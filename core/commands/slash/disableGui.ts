import { SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";
// import {diableGui} from "../../../extensions/vscode/src/autocomplete/completionProvider.js"
// import * as vscode from "vscode";

const DisableGuiMethodCommand: SlashCommand = {
  name: "disableGui",
  description: "disable the conpletion Gui method",
  run: async function* ({ input }) {
    Ollama.disableGui();
    // diableGui();
    console.log(input);
  },
};

export default DisableGuiMethodCommand;