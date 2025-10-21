from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from flask import Flask, request, jsonify, Response

model_path="/home/shape_model/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# data = request.json
data = {
    'content' : 'print("hello")'
}
content = data['content']

datastore_path = './datastore_customized_repo.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512 * 1024 * 1024,
    vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
)

token_list = tokenizer.encode(content,truncation=True)
writer.add_entry(token_list)
writer.add_entry(token_list)
writer.finalize()

writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512 * 1024 * 1024,
    vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
)

token_list = tokenizer.encode(content,truncation=True)
writer.add_entry(token_list)

writer.finalize()
