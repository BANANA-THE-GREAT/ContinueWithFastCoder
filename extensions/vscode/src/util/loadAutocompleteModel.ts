import { ConfigHandler } from "core/config/ConfigHandler";
import Ollama from "core/llm/llms/Ollama";
import { GlobalContext } from "core/util/GlobalContext";
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { VsCodeIde } from "../VsCodeIde";
import { spawn } from 'child_process';
import axios from 'axios';

import type { ILLM } from "core";

export class TabAutocompleteModel {
  private _llm: ILLM | undefined;
  private defaultTag = "qwen2.5-coder:1.5b";
  private globalContext: GlobalContext = new GlobalContext();

  constructor(private configHandler: ConfigHandler, private ide: VsCodeIde) {}

  clearLlm() {
    this._llm = undefined;
  }

  async getDefaultTabAutocompleteModel() {
    const llm = new Ollama({
      model: "deepseek-coder:6.7b",
    });
    await this.get_datastore_repo(llm);

    // const llm = new Ollama({
    //   model: this.defaultTag,
    // });

    // try {
    //   const models = await llm.listModels();
    //   if (!models.includes(this.defaultTag)) {
    //     return undefined;
    //   }
    // } catch (e) {
    //   return undefined;
    // }

    return llm;
  }

  // 判断是否为Python文件
  isPythonFile(file: string): boolean {
    return file.endsWith('.py');
  }

  // 递归查找目录中的所有Python文件
  getPythonFilesInDir(dir: string): string[] {
    let pythonFiles: string[] = [];
    
    // 读取目录内容
    const files = fs.readdirSync(dir);
    
    files.forEach((file) => {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        // 如果是目录，则递归
        pythonFiles = pythonFiles.concat(this.getPythonFilesInDir(fullPath));
      } else if (stat.isFile() && this.isPythonFile(file)) {
        // 如果是Python文件，则加入列表
        pythonFiles.push(fullPath);
      }
    });
    
    return pythonFiles;
  }

  getWorkspaceRoot(): string | undefined {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        return workspaceFolders[0].uri.fsPath;  // 获取第一个工作区的根目录路径
    }
    return undefined;
  }

  async sendData(url: string, data: string) {
    try {
      const response = await fetch(url, {
        method: "POST", // 请求方法
        headers: {
          "Content-Type": "application/json", // 设置请求体为 JSON 格式
        },
        body: JSON.stringify({"content" : data}), // 将数据转为 JSON 字符串
      });
  
      // 处理响应
      if (!response.ok) {
        throw new Error("Network response was not ok.");
      }
  
      const responseData = await response.json();
      console.log("Response from backend:", responseData);
    } catch (error) {
      console.error("Error sending data:", error);
    }
  }

  async get_datastore_repo(llm: Ollama) {
    // 处理工作区中的所有目录
    const dir = this.getWorkspaceRoot();
    if (!dir) {
      return;
    }
    const pythonFiles = this.getPythonFilesInDir(dir);

    const response1 = await llm.fetch(llm.getEndpoint("init_writer"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    console.log('Response from Python:', response1);

    if (pythonFiles.length > 0) {
      for (const file of pythonFiles) {
        try {
          // 读取文件内容
          const fileContent = fs.readFileSync(file, 'utf-8');
          
          // 执行异步操作，等待其完成
          const response2 = await llm.fetch(llm.getEndpoint("append_writer"), {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ "content": fileContent }),
          });
          
          console.log('Response from Python:', response2);
        } catch (error) {
          console.error(`Error reading file ${file}:`, error);
        }
      }
    }

    const response3 = await llm.fetch(llm.getEndpoint("finalize_writer"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    console.log('Response from Python:', response3);
  }

  async get() {
    if (!this._llm) {
      const { config } = await this.configHandler.loadConfig();
      if (!config) {
        return undefined;
      }

      // if (config.tabAutocompleteModels?.length) {
      //   const selected = this.globalContext.get("selectedTabAutocompleteModel");
      //   if (selected) {
      //     this._llm =
      //       config.tabAutocompleteModels?.find(
      //         (model) => model.title === selected,
      //       ) ?? config.tabAutocompleteModels?.[0];
      //   } else {
      //     if (config.tabAutocompleteModels[0].title) {
      //       this.globalContext.update(
      //         "selectedTabAutocompleteModel",
      //         config.tabAutocompleteModels[0].title,
      //       );
      //     }
      //     this._llm = config.tabAutocompleteModels[0];
      //   }
      // } else {
        this._llm = await this.getDefaultTabAutocompleteModel();
      // }
    }

    return this._llm;
  }
}
