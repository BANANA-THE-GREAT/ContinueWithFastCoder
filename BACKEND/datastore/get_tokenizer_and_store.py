from transformers import AutoTokenizer

model_path = "/home/shape_model/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.save_pretrained('tokenizer_path')

tokenizer = AutoTokenizer.from_pretrained('tokenizer_path')
