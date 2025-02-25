import { CompletionProvider } from "core/autocomplete/CompletionProvider";
import {
  type AutocompleteInput,
  type AutocompleteOutcome,
} from "core/autocomplete/util/types";
import { ConfigHandler } from "core/config/ConfigHandler";
import * as URI from "uri-js";
import { v4 as uuidv4 } from "uuid";
import * as vscode from "vscode";

import { showFreeTrialLoginMessage } from "../util/messages";
import { VsCodeWebviewProtocol } from "../webviewProtocol";

import { getDefinitionsFromLsp } from "./lsp";
import { RecentlyEditedTracker } from "./recentlyEdited";
import { RecentlyVisitedRangesService } from "./RecentlyVisitedRangesService";
import {
  StatusBarStatus,
  getStatusBarStatus,
  setupStatusBar,
  stopStatusBarLoading,
} from "./statusBar";

import type { TabAutocompleteModel } from "../util/loadAutocompleteModel";
import { startLocalOllama } from "core/util/ollamaHelper";
import type { IDE } from "core";

import Ollama from "core/llm/llms/Ollama";
import { useGui } from "../commands";
import * as fs from 'fs';
import * as path from 'path';
import { C } from "core/autocomplete/constants/AutocompleteLanguageInfo";

const Diff = require("diff");
const decoratedRanges = new Map<vscode.TextEditor, vscode.Range[]>();

export const decorationTypeForCache = vscode.window.createTextEditorDecorationType({
  fontStyle: 'italic', // 斜体
  color: 'rgba(255, 166, 0, 0.75)', // 可选的颜色
});

export const decorationTypeForStore = vscode.window.createTextEditorDecorationType({
  fontStyle: 'italic', // 斜体
  color: 'rgba(255, 17, 0, 0.83)', // 可选的颜色
});

export const decorationTypeForModel = vscode.window.createTextEditorDecorationType({
  fontStyle: 'italic', // 斜体
  color: 'rgba(0, 255, 179, 0.75)', // 可选的颜色
});

export let cacheRanges: { [filepath: string]: vscode.Range[] } = {};
export let storeRanges: { [filepath: string]: vscode.Range[] } = {};
export let modelRanges: { [filepath: string]: vscode.Range[] } = {};
let guiCacheRanges: { [filepath: string]: vscode.Range[] } = {};
let guiStoreRanges: { [filepath: string]: vscode.Range[] } = {};
let guiModelRanges: { [filepath: string]: vscode.Range[] } = {};
export let curFilePath: string = '';
let group = 0;

interface DiffType {
  count: number;
  added: boolean;
  removed: boolean;
  value: string;
}

interface VsCodeCompletionInput {
  document: vscode.TextDocument;
  position: vscode.Position;
  context: vscode.InlineCompletionContext;
}

export class ContinueCompletionProvider
  implements vscode.InlineCompletionItemProvider
{
  private onError(e: any) {
    const options = ["Documentation"];
    if (e.message.includes("Ollama may not be installed")) {
      options.push("Download Ollama");
    } else if (e.message.includes("Ollama may not be running")) {
      options.unshift("Start Ollama"); // We want "Start" to be the default choice
    }

    if (e.message.includes("Please sign in with GitHub")) {
      showFreeTrialLoginMessage(
        e.message,
        this.configHandler.reloadConfig.bind(this.configHandler),
        () => {
          void this.webviewProtocol.request("openOnboardingCard", undefined);
        },
      );
      return;
    }
    vscode.window.showErrorMessage(e.message, ...options).then((val) => {
      if (val === "Documentation") {
        vscode.env.openExternal(
          vscode.Uri.parse(
            "https://docs.continue.dev/features/tab-autocomplete",
          ),
        );
      } else if (val === "Download Ollama") {
        vscode.env.openExternal(vscode.Uri.parse("https://ollama.ai/download"));
      } else if (val == "Start Ollama") {
        startLocalOllama(this.ide);
      }
    });
  }

  private completionProvider: CompletionProvider;
  private recentlyVisitedRanges: RecentlyVisitedRangesService;
  private recentlyEditedTracker = new RecentlyEditedTracker();

  constructor(
    private readonly configHandler: ConfigHandler,
    private readonly ide: IDE,
    private readonly tabAutocompleteModel: TabAutocompleteModel,
    private readonly webviewProtocol: VsCodeWebviewProtocol,
  ) {
    this.completionProvider = new CompletionProvider(
      this.configHandler,
      this.ide,
      this.tabAutocompleteModel.get.bind(this.tabAutocompleteModel),
      this.onError.bind(this),
      getDefinitionsFromLsp,
    );
    this.recentlyVisitedRanges = new RecentlyVisitedRangesService(ide);
  }

  _lastShownCompletion: AutocompleteOutcome | undefined;

  _lastVsCodeCompletionInput: VsCodeCompletionInput | undefined;

  public async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
    //@ts-ignore
  ): ProviderResult<InlineCompletionItem[] | InlineCompletionList> {
    const enableTabAutocomplete =
      getStatusBarStatus() === StatusBarStatus.Enabled;
    if (token.isCancellationRequested || !enableTabAutocomplete) {
      return null;
    }

    if (document.uri.scheme === "vscode-scm") {
      return null;
    }

    // Don't autocomplete with multi-cursor
    const editor = vscode.window.activeTextEditor;
    if (editor && editor.selections.length > 1) {
      return null;
    }

    // If the text at the range isn't a prefix of the intellisense text,
    // no completion will be displayed, regardless of what we return
    if (
      context.selectedCompletionInfo &&
      !context.selectedCompletionInfo.text.startsWith(
        document.getText(context.selectedCompletionInfo.range),
      )
    ) {
      return null;
    }

    let injectDetails: string | undefined = undefined;

    // The first time intellisense dropdown shows up, and the first choice is selected,
    // we should not consider this. Only once user explicitly moves down the list
    const newVsCodeInput = {
      context,
      document,
      position,
    };
    const selectedCompletionInfo = context.selectedCompletionInfo;
    this._lastVsCodeCompletionInput = newVsCodeInput;

    try {
      const abortController = new AbortController();
      const signal = abortController.signal;
      token.onCancellationRequested(() => abortController.abort());

      // Handle notebook cells
      const pos = {
        line: position.line,
        character: position.character,
      };
      let manuallyPassFileContents: string | undefined = undefined;
      if (document.uri.scheme === "vscode-notebook-cell") {
        const notebook = vscode.workspace.notebookDocuments.find((notebook) =>
          notebook
            .getCells()
            .some((cell) =>
              URI.equal(cell.document.uri.toString(), document.uri.toString()),
            ),
        );
        if (notebook) {
          const cells = notebook.getCells();
          manuallyPassFileContents = cells
            .map((cell) => {
              const text = cell.document.getText();
              if (cell.kind === vscode.NotebookCellKind.Markup) {
                return `"""${text}"""`;
              } else {
                return text;
              }
            })
            .join("\n\n");
          for (const cell of cells) {
            if (
              URI.equal(cell.document.uri.toString(), document.uri.toString())
            ) {
              break;
            } else {
              pos.line += cell.document.getText().split("\n").length + 1;
            }
          }
        }
      }

      // Manually pass file contents for unsaved, untitled files
      if (document.isUntitled) {
        manuallyPassFileContents = document.getText();
      }

      // Handle commit message input box
      let manuallyPassPrefix: string | undefined = undefined;

      const input: AutocompleteInput = {
        pos,
        manuallyPassFileContents,
        manuallyPassPrefix,
        selectedCompletionInfo,
        injectDetails,
        isUntitledFile: document.isUntitled,
        completionId: uuidv4(),
        filepath: document.uri.toString(),
        recentlyVisitedRanges: this.recentlyVisitedRanges.getSnippets(),
        recentlyEditedRanges:
          await this.recentlyEditedTracker.getRecentlyEditedRanges(),
      };

      setupStatusBar(undefined, true);
      const outcome =
        await this.completionProvider.provideInlineCompletionItems(
          input,
          signal,
        );

      if (!outcome || !outcome.completion) {
        return null;
      }

      // VS Code displays dependent on selectedCompletionInfo (their docstring below)
      // We should first always make sure we have a valid completion, but if it goes wrong we
      // want telemetry to be correct
      /**
       * Provides information about the currently selected item in the autocomplete widget if it is visible.
       *
       * If set, provided inline completions must extend the text of the selected item
       * and use the same range, otherwise they are not shown as preview.
       * As an example, if the document text is `console.` and the selected item is `.log` replacing the `.` in the document,
       * the inline completion must also replace `.` and start with `.log`, for example `.log()`.
       *
       * Inline completion providers are requested again whenever the selected item changes.
       */
      if (selectedCompletionInfo) {
        outcome.completion = selectedCompletionInfo.text + outcome.completion;
      }
      outcome.completion = "<｜c> main():<c｜>Th<｜m>is is a sample text<m｜> for <｜d>wrapping<d｜>";
      const willDisplay = this.willDisplay(
        document,
        selectedCompletionInfo,
        signal,
        outcome,
      );
      if (!willDisplay) {
        return null;
      }

      // Mark displayed
      this.completionProvider.markDisplayed(input.completionId, outcome);
      this._lastShownCompletion = outcome;

      // Construct the range/text to show
      const startPos = selectedCompletionInfo?.range.start ?? position;
      let range = new vscode.Range(startPos, startPos);
      let completionText = outcome.completion;
      const isSingleLineCompletion = outcome.completion.split("\n").length <= 1;

      if (isSingleLineCompletion) {
        const lastLineOfCompletionText = completionText.split("\n").pop();
        const currentText = document
          .lineAt(startPos)
          .text.substring(startPos.character);
        const diffs: DiffType[] = Diff.diffWords(
          currentText,
          lastLineOfCompletionText,
        );

        if (diffPatternMatches(diffs, ["+"])) {
          // Just insert, we're already at the end of the line
        } else if (
          diffPatternMatches(diffs, ["+", "="]) ||
          diffPatternMatches(diffs, ["+", "=", "+"])
        ) {
          // The model repeated the text after the cursor to the end of the line
          range = new vscode.Range(
            startPos,
            document.lineAt(startPos).range.end,
          );
        } else if (
          diffPatternMatches(diffs, ["+", "-"]) ||
          diffPatternMatches(diffs, ["-", "+"])
        ) {
          // We are midline and the model just inserted without repeating to the end of the line
          // We want to move the cursor to the end of the line
          // range = new vscode.Range(
          //   startPos,
          //   document.lineAt(startPos).range.end,
          // );
          // // Find the last removed part of the diff
          // const lastRemovedIndex = findLastIndex(
          //   diffs,
          //   (diff) => diff.removed === true,
          // );
          // const lastRemovedContent = diffs[lastRemovedIndex].value;
          // completionText += lastRemovedContent;
        } else {
          // Diff is too complicated, just insert the first added part of the diff
          // This is the safe way to ensure that it is displayed
          if (diffs[0]?.added) {
            completionText = diffs[0].value;
          } else {
            // If the first part of the diff isn't an insertion, then the model is
            // probably rewriting other parts of the line
            return undefined;
          }
        }
      } else {
        // Extend the range to the end of the line for multiline completions
        range = new vscode.Range(startPos, document.lineAt(startPos).range.end);
      }

      // completionText = "<｜c> main():<c｜>";
      let processedText = completionText.replace(/<｜c>|<c｜>|<｜d>|<d｜>|<｜m>|<m｜>/g, '');
      const completionItem = new vscode.InlineCompletionItem(
        processedText,
        range,
        {
          title: "Log Autocomplete Outcome",
          command: "continue.logAutocompleteOutcome",
          arguments: [input.completionId, this.completionProvider],
        },
      );
      curFilePath = outcome.filepath;

      //////////////////////////////////////////////
      const editor = vscode.window.activeTextEditor;
      const workspaceFolder = vscode.workspace.workspaceFolders;
      let FilePath = '';
      if (workspaceFolder && workspaceFolder.length > 0) {
        FilePath = workspaceFolder[0].uri.fsPath;
      } else {
        console.log("there is no workspace dir");
      }
      const jsonlFilename = `.autocomplete.jsonl`; // 添加 .jsonl 扩展名
      const autocompleteFilePath = path.join(FilePath, jsonlFilename);
      let autocompleteResults:any = [];
      // cacheRanges = [];
      // modelRanges = [];
      // storeRanges = [];
      // const autocompleteResults = autocompleteData.split('\n').map(line => JSON.parse(line));
      // curFilePath = outcome.filepath;
      if (!cacheRanges[curFilePath]) {
        cacheRanges[curFilePath] = [];
        modelRanges[curFilePath] = [];
        storeRanges[curFilePath] = [];
      }
      const curText = editor?.document.getText();
      fs.readFile(autocompleteFilePath, 'utf-8', (err, data) => {
        if (err) {
            console.error('读取文件时发生错误:', err);
            return;
        }
        // autocompleteResults = data.split('\n').map(line => JSON.parse(line));
        const lines = data.split('\n');
        // console.log("hello");
        // 解析每一行为 JSON 对象
        for (let line of lines) {
            line = line.trim(); // 去除首尾空格
            if (line) { // 确保不是空行
                try {
                    const json = JSON.parse(line);
                    autocompleteResults.push(json);
                } catch (e) {
                    vscode.window.showWarningMessage(`Failed to parse line: ${line}`);
                }
            }
        }

        // 输出结果（调试用）
        // console.log(autocompleteResults);
        autocompleteResults = autocompleteResults.filter((item:any) =>
          item.filepath === curFilePath && item.accepted === true
        );

        const matchingRanges: vscode.Range[] = [];
        for (const item of autocompleteResults) {
          if (curText && editor && curText.startsWith(item.fullPrefix)) {
            const textAfterPrefix = curText.slice(item.fullPrefix.length);
            const filteredCompletion = item.completion.replace(/<｜c>|<c｜>|<｜d>|<d｜>|<｜m>|<m｜>/g, '');
            if (textAfterPrefix.startsWith(filteredCompletion)) {
              const start0 = editor.document.positionAt(item.fullPrefix.length);
              const end0 = editor.document.positionAt(item.fullPrefix.length + filteredCompletion.length);
              const range = new vscode.Range(start0, end0);
              const isDuplicate = matchingRanges.some(r =>
                 r.start.isEqual(range.start) && r.end.isEqual(range.end)
              );
              if (!isDuplicate) {
                matchingRanges.push(range);
              }

              const completionWithTags = item.completion;
              let start = editor.document.positionAt(item.fullPrefix.length);
              let end = start;
              let distance = 0;
              for (let i = 0; i < completionWithTags.length; i++) {
                const char = completionWithTags[i];
                const nextChar = completionWithTags[i + 1];
                const nextNextChar = completionWithTags[i + 2];
                const nextNextNextChar = completionWithTags[i + 3];
                if (char === '<' && nextChar === '｜' && nextNextChar === 'c' && nextNextNextChar === '>') {
                  group = 1;
                  i += 3;
                } else if (char === '<' && nextChar === '｜' && nextNextChar === 'd' && nextNextNextChar === '>') {
                  group = 2;
                  i += 3;
                } else if (char === '<' && nextChar === '｜' && nextNextChar === 'm' && nextNextNextChar === '>') {
                  group = 3;
                  i += 3;
                } else if (char === '<' && (nextChar === 'c' || nextChar === 'd' || nextChar === 'm') && nextNextChar === '｜' && nextNextNextChar === '>') {
                  end = editor.document.positionAt(editor.document.offsetAt(start) + distance);
                  const range = new vscode.Range(start, end);
                  if (group === 1) {
                    if (!cacheRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      cacheRanges[curFilePath].push(range);
                    }
                  } else if (group === 2) {
                    if (!storeRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      storeRanges[curFilePath].push(range);
                    }
                  } else if (group === 3) {
                    if (!modelRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      modelRanges[curFilePath].push(range);
                    }
                  }
                  start = end;
                  distance = 0;
                  i += 3;
                } else {
                  distance++;
                }
              }
              if (distance != 0) {
                end = editor.document.positionAt(editor.document.offsetAt(start) + distance);
                const range = new vscode.Range(start, end);
                  if (group === 1) {
                    if (!cacheRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      cacheRanges[curFilePath].push(range);
                    }
                  } else if (group === 2) {
                    if (!storeRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      storeRanges[curFilePath].push(range);
                    }
                  } else if (group === 3) {
                    if (!modelRanges[curFilePath].some(r => r.start.isEqual(range.start) && r.end.isEqual(range.end))) {
                      modelRanges[curFilePath].push(range);
                    }
                  }
                start = end;
                distance = 0;
              }
            }
          }
        }

        // 输出匹配的 range（调试用）
        // console.log(matchingRanges);

        // if (editor) {
        //   if (useGui) {
        //     // editor.setDecorations(decorationTypeForCache, matchingRanges);
        //     editor.setDecorations(decorationTypeForCache, cacheRanges[curFilePath]);
        //     editor.setDecorations(decorationTypeForModel, modelRanges[curFilePath]);
        //     editor.setDecorations(decorationTypeForStore, storeRanges[curFilePath]);
        //   } else {
        //     editor.setDecorations(decorationTypeForCache, []);
        //     editor.setDecorations(decorationTypeForModel, []);
        //     editor.setDecorations(decorationTypeForStore, []);
        //   }
        // }
      }
    );

      // const autocompleteData = fs.readFileSync(autocompleteFilePath, 'utf-8');


      //////////////////////////////////////////////

      (completionItem as any).completeBracketPairs = true;
      return [completionItem];
    } finally {
      stopStatusBarLoading();
    }
  }

  willDisplay(
    document: vscode.TextDocument,
    selectedCompletionInfo: vscode.SelectedCompletionInfo | undefined,
    abortSignal: AbortSignal,
    outcome: AutocompleteOutcome,
  ): boolean {
    if (selectedCompletionInfo) {
      const { text, range } = selectedCompletionInfo;
      if (!outcome.completion.startsWith(text)) {
        console.log(
          `Won't display completion because text doesn't match: ${text}, ${outcome.completion}`,
          range,
        );
        return false;
      }
    }

    if (abortSignal.aborted) {
      return false;
    }

    return true;
  }
}

function getDecorationRanges(document: vscode.TextDocument, matchingRanges: vscode.Range[]): vscode.DecorationOptions[] {
  const decorations: vscode.DecorationOptions[] = [];
  const lineCount = document.lineCount;

  for (let i = 0; i < lineCount; i++) {
      const line = document.lineAt(i);
      const range = new vscode.Range(line.range.start, line.range.end);
      decorations.push({ range });
  }

  return decorations;
}

type DiffPartType = "+" | "-" | "=";

function diffPatternMatches(
  diffs: DiffType[],
  pattern: DiffPartType[],
): boolean {
  if (diffs.length !== pattern.length) {
    return false;
  }

  for (let i = 0; i < diffs.length; i++) {
    const diff = diffs[i];
    const diffPartType: DiffPartType =
      !diff.added && !diff.removed ? "=" : diff.added ? "+" : "-";

    if (diffPartType !== pattern[i]) {
      return false;
    }
  }

  return true;
}

// function processAutocompleteResults():void {

// }

export function enable(range: vscode.Range): void {
  const editor = vscode.window.activeTextEditor;
  // processAutocompleteResults();
  // curFilePath =
  if (editor) {
    const filteredCacheRanges = filterRangesByIntersection(cacheRanges[curFilePath], range);
    const filteredModelRanges = filterRangesByIntersection(modelRanges[curFilePath], range);
    const filteredStoreRanges = filterRangesByIntersection(storeRanges[curFilePath], range);

    guiCacheRanges[curFilePath] = mergeRanges(guiCacheRanges[curFilePath] || [], filteredCacheRanges);
    guiModelRanges[curFilePath] = mergeRanges(guiModelRanges[curFilePath] || [], filteredModelRanges);
    guiStoreRanges[curFilePath] = mergeRanges(guiStoreRanges[curFilePath] || [], filteredStoreRanges);

    editor.setDecorations(decorationTypeForCache, guiCacheRanges[curFilePath]);
    editor.setDecorations(decorationTypeForModel, guiModelRanges[curFilePath]);
    editor.setDecorations(decorationTypeForStore, guiStoreRanges[curFilePath]);
  }
}

function mergeRanges(existingRanges: vscode.Range[], newRanges: vscode.Range[]): vscode.Range[] {
  const allRanges = [...existingRanges, ...newRanges];
  const sortedRanges = allRanges.sort((a, b) => a.start.compareTo(b.start));

  const mergedRanges: vscode.Range[] = [];
  for (const range of sortedRanges) {
    if (mergedRanges.length === 0) {
      mergedRanges.push(range);
    } else {
      const lastRange = mergedRanges[mergedRanges.length - 1];
      if (range.start.isBefore(lastRange.end)) {
        // 如果当前范围与最后一个合并范围有重叠，合并它们
        mergedRanges[mergedRanges.length - 1] = new vscode.Range(
          lastRange.start,
          range.end.isAfter(lastRange.end) ? range.end : lastRange.end
        );
      } else {
        // 否则，添加当前范围到合并范围数组
        mergedRanges.push(range);
      }
    }
  }

  return mergedRanges;
}

function filterRangesByIntersection(ranges: vscode.Range[], targetRange: vscode.Range): vscode.Range[] {
  return ranges.map(range => {
    const intersection = range.intersection(targetRange);
    if (!intersection) return undefined;

    const start = range.start.isBefore(targetRange.start) ? targetRange.start : range.start;
    const end = range.end.isAfter(targetRange.end) ? targetRange.end : range.end;

    return new vscode.Range(start, end);
  }).filter(range => range !== undefined) as vscode.Range[]; // 过滤掉 undefined
}

export function disable(range: vscode.Range): void {
  const editor = vscode.window.activeTextEditor;
  // processAutocompleteResults();
  if (!editor) return;

  guiCacheRanges[curFilePath] = removeIntersectingRanges(guiCacheRanges[curFilePath] || [], range);
  guiModelRanges[curFilePath] = removeIntersectingRanges(guiModelRanges[curFilePath] || [], range);
  guiStoreRanges[curFilePath] = removeIntersectingRanges(guiStoreRanges[curFilePath] || [], range);

  editor.setDecorations(decorationTypeForCache, guiCacheRanges[curFilePath]);
  editor.setDecorations(decorationTypeForModel, guiModelRanges[curFilePath]);
  editor.setDecorations(decorationTypeForStore, guiStoreRanges[curFilePath]);
}

function removeIntersectingRanges(existingRanges: vscode.Range[], rangeToRemove: vscode.Range): vscode.Range[] {
  return existingRanges.flatMap(existingRange => {
    const intersection = existingRange.intersection(rangeToRemove);
    if (!intersection) {
      return [existingRange];
    }

    const resultRanges: vscode.Range[] = [];

    if (existingRange.start.isBefore(rangeToRemove.start)) {
      resultRanges.push(new vscode.Range(existingRange.start, rangeToRemove.start));
    }

    if (existingRange.end.isAfter(rangeToRemove.end)) {
      resultRanges.push(new vscode.Range(rangeToRemove.end, existingRange.end));
    }

    return resultRanges;
  });
}

export function disableAll(): void {
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    editor.setDecorations(decorationTypeForCache, []);
    editor.setDecorations(decorationTypeForModel, []);
    editor.setDecorations(decorationTypeForStore, []);
  }
}

export function enableAll(): void {
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    editor.setDecorations(decorationTypeForCache, cacheRanges[curFilePath]);
    editor.setDecorations(decorationTypeForModel, modelRanges[curFilePath]);
    editor.setDecorations(decorationTypeForStore, storeRanges[curFilePath]);
  }
}
