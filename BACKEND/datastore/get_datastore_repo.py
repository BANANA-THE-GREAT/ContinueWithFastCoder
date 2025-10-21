from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from tqdm import tqdm
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    default="/home/shape_model/deepseek-coder-6.7b-base",
    # default="/home/shape_model/CodeLlama-7b-instruct-hf",
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

base_dir = '/home/jiaoziqian/LLMAcceleration-REST/Source_Code/'
topics=os.listdir(base_dir)
for topic in topics:
    if not os.path.exists(f'/home/jiaoziqian/LLMAcceleration-REST/datastore/repo_datastore/{topic}'):
        os.mkdir(f'/home/jiaoziqian/LLMAcceleration-REST/datastore/repo_datastore/{topic}')
    
            
    repos=os.listdir(base_dir+topic)
    for repo in repos:
        # if repo=='PySimpleSOAP':
        directory_path = f'{base_dir}/{topic}/{repo}'
        py_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))


        datastore_path = f'/home/jiaoziqian/LLMAcceleration-REST/datastore/repo_datastore/{topic}/datastore_repo_{repo}.idx'
        writer = draftretriever.Writer(
            index_file_path=datastore_path,
            max_chunk_len=512 * 1024 * 1024,
            vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
        )

        total_length = len(py_files)
        print(f'datastore_repo_{repo}')
        print("number of files: ", total_length)

        for file in tqdm(py_files, total=len(py_files)):
            content = open(file,'r',encoding='utf-8',errors='ignore').read()
            token_list = tokenizer.encode(content,truncation=True)
            writer.add_entry(token_list)

        # print('end')
        writer.finalize()