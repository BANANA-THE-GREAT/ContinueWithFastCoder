import { IDE, SlashCommand } from "../../index.js";
import Ollama from "../../llm/llms/Ollama.js";
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from "node:url";

const llm = new Ollama({
  model: "deepseek-coder:6.7b",
});

// 判断是否为Python文件
function isPythonFile(file: string): boolean {
  return file.endsWith('.py');
}

// 递归查找目录中的所有Python文件
function getPythonFilesInDir(dir: string): string[] {
  let pythonFiles: string[] = [];
  
  // 读取目录内容
  const files = fs.readdirSync(dir);
  
  files.forEach((file) => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      // 如果是目录，则递归
      pythonFiles = pythonFiles.concat(getPythonFilesInDir(fullPath));
    } else if (stat.isFile() && isPythonFile(file)) {
      // 如果是Python文件，则加入列表
      pythonFiles.push(fullPath);
    }
  });
  
  return pythonFiles;
}

async function getWorkspaceRoot(ide: IDE): Promise<string | undefined> {
  const workspaceDirs = await ide.getWorkspaceDirs();
  const workspaceDirectory = workspaceDirs?.[0] || "";  // 获取第一个工作区的根目录路径
  const root = fileURLToPath(workspaceDirectory);
  if (root) {
      return root;
  }
  return undefined;
}

async function get_datastore_repo(ide: IDE) {
  ide.showToast("info", "Begin to renew repo datastore ...");
  // ide.showToast("info", "正在更新检索库 ...");
  // 处理工作区中的所有目录
  const dir = await getWorkspaceRoot(ide);
  if (!dir) {
    return;
  }
  const pythonFiles = getPythonFilesInDir(dir);

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
        console.log(file)
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

  ide.showToast("info", "Repo datastore renewed successfully!");
  // ide.showToast("info", "检索库更新完成");
}

const RenewRepoDatastoreCommand: SlashCommand = {
  name: "renew",
  description: "renew the repo datastore",
  run: async function* ({ ide, input }) {
    get_datastore_repo(ide);

    console.log(input);
  },
};

export default RenewRepoDatastoreCommand;
