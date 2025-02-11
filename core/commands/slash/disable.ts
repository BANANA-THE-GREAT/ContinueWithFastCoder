import { SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";

const DisableAccelerationMethodCommand: SlashCommand = {
  name: "disableAcc",
  description: "disable the acceleration method",
  run: async function* ({ input }) {
    Ollama.disableAcc();
    console.log(input);
  },
};

export default DisableAccelerationMethodCommand;