# This code is adapted from https://github.com/FasterDecoding/REST;
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from tqdm import tqdm
import argparse
import json
from collections import Counter
from nltk.util import ngrams
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    default="/home/shape_model/deepseek-coder-6.7b-base",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
# parser.add_argument(
#     "--large-datastore",
#     type=bool,
#     default=False,
#     help="Whether to use a large datastore",
# )
args = parser.parse_args()
print(args)


tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# segment = 30 if args.large_datastore else 1 # Maximum number of segment: 144
# data_files = []
# for i in range(segment):
#     if i>=100:
#         data_files.append(f"data-00{i}-of-00144.parquet")
#     elif i >=10:
#         data_files.append(f"data-000{i}-of-00144.parquet")
#     else:
#         data_files.append(f"data-0000{i}-of-00144.parquet")
# print("data_files:", data_files)

# dataset = load_dataset('the-stack-dedup', \
#     data_dir='data/python', split='train', data_files=data_files)


data=[]
f=open("/home/zhaoqianhui/REST/dev_eval/data/ngrams_repo.jsonl",'r',encoding='utf-8',errors='ignore')
for l in f.readlines():
    data.append(json.loads(l))
f.close()

for i in tqdm(data,total=len(data)):
    topic=i['topic']
    repo=i['repo']
    sequences=[j['sequence'] for j in i['sequences']]

    if not os.path.exists(f'/home/zhaoqianhui/REST/datastore/repo_ngram_datastore/{topic}'):
        os.mkdir(f'/home/zhaoqianhui/REST/datastore/repo_ngram_datastore/{topic}')
    datastore_path = f'/home/zhaoqianhui/REST/datastore/repo_ngram_datastore/{topic}/datastore_repo_ngram_{repo}.idx'

    writer = draftretriever.Writer(
        index_file_path=datastore_path,
        max_chunk_len=512 * 1024 * 1024,
        vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
    )
    
    for s in tqdm(sequences, total=len(sequences)):
        token_list = tokenizer.encode(s)
        writer.add_entry(token_list)

    writer.finalize()
