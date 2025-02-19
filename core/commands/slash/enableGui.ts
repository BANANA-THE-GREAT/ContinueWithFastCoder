import { SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";
// import * as vscode from "vscode";
// import {enableGui} from "../../../extensions/vscode/src/autocomplete/completionProvider.js"


const EnableGuiMethodCommand: SlashCommand = {
  name: "enableGui",
  description: "enable the completion Gui method",
  run: async function* ({ input }) {
    Ollama.enableGui();
    // enableGui();
    console.log(input);
  },
};

export default EnableGuiMethodCommand;