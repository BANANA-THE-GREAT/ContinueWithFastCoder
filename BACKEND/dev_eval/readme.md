## 文件说明

**baseline_test.py:** 自回归

**dev_eval_test.py:** 使用datastore生成

**bayesian_optimization.py:** 贝叶斯优化尝试

**get_output.py:** 保存检索结果

**cache_test.py:** 使用cache生成，使用了修改后的util（`rest/model/my_utils.py`中包含了对原`utils.py`文件更新的内容，其中`generate_draft_buffers`是根据draft_choices生成tree_attention的过程，需要注意它的两个参数含义，已在注释中注明）