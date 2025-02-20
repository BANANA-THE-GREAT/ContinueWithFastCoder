import fs from "fs";
import path from "path";

import { getDevDataFilePath } from "./paths.js";

export function logDevData(tableName: string, data: any, FilePath: string) {
  const filepath0: string = getDevDataFilePath(tableName);
  const jsonLine = JSON.stringify(data);
  fs.writeFileSync(filepath0, `${jsonLine}\n`, { flag: "a" });
  if (FilePath != '') {
    let filepath = FilePath;

    // const filepath_temp: string = data.filepath;
    // const filepath = filepath_temp.replace("file:///", "").replace("%3A", ":")
    // const filename = path.basename(filepath, path.extname(filepath)); // 提取文件名（去掉扩展名）
    const jsonlFilename = `.autocomplete.jsonl`; // 添加 .jsonl 扩展名
    const jsonlFilePath = path.join(filepath, jsonlFilename);
    const directory = path.dirname(jsonlFilePath);
    if (!fs.existsSync(directory)) {
      fs.mkdirSync(directory, { recursive: true });
    }
    fs.writeFileSync(jsonlFilePath, `${jsonLine}\n`, { flag: "a" });
    console.log(`数据已追加到文件：${jsonlFilePath}`);
  }
  
}
