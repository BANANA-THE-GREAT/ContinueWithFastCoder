# 后端说明：

VSCode插件后端部分使用Flask包装当前论文算法，提供流式输出的接口。

核心文件：`dev_eval/final test.py`；
运行命令:
``` shell
cd dev_eval/
RAYON_NUM_THREADS=6 CUDA VISIBLE DEVICES=x python3 final test.py  # x 为显卡编号。
```
