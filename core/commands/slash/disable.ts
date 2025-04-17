import { SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";

const DisableAccelerationMethodCommand: SlashCommand = {
  name: "disableAcc",
  description: "disable the acceleration method",
  run: async function* ({ ide, input }) {
    Ollama.disableAcc();
    console.log(input);
    // ide.showToast("info", "Acceleration method disabled successfully");
    ide.showToast("info", "加速方法已关闭");
  },
};

export default DisableAccelerationMethodCommand;