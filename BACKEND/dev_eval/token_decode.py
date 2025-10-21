import sys
import torch
sys.path.append("/home/jiaoziqian/REST/")
from rest.model.rest_model import RestModel

model = RestModel.from_pretrained(
        "/home/shape_model/CodeLlama-7b-instruct-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
tokenizer = model.get_tokenizer()

vocab = tokenizer.get_vocab()
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, token_id in vocab.items():
        decoded_token = tokenizer.decode([token_id])
        # decoded_token = decoded_token.replace("\ufffd", " ")
        f.write(f"{token_id}\t{decoded_token}\n")


# token_id = 1
# token_text = tokenizer.decode([token_id], skip_special_tokens=True)
# print(f"Token ID {token_id} corresponds to: {token_text}")

# token_id = 957
# token_text = tokenizer.decode([token_id], skip_special_tokens=True)
# print(f"Token ID {token_id} corresponds to: {token_text}")