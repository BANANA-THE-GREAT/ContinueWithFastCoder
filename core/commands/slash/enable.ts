import { SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";

const EnableAccelerationMethodCommand: SlashCommand = {
  name: "enableAcc",
  description: "enable the acceleration method",
  run: async function* ({ ide, input }) {
    Ollama.enableAcc();
    console.log(input);
    ide.showToast("info", "Acceleration method enabled successfully");
    // ide.showToast("info", "加速方法已开启");
  },
};

export default EnableAccelerationMethodCommand;