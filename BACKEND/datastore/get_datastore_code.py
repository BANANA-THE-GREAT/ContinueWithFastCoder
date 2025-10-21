from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
import draftretriever
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    # default="/home/shape_model/Qwen2.5-Coder-1.5B",
    default="/home/shape_model/CodeLlama-7b-instruct-hf",
    # default="/home/shape_model/deepseek-coder-6.7b-base",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--large-datastore",
    type=bool,
    default=False,
    help="Whether to use a large datastore",
)
args = parser.parse_args()
print(args)

config = AutoConfig.from_pretrained(args.model_path)
print(f"Model's maximum sequence length: {config.max_position_embeddings}")


tokenizer = AutoTokenizer.from_pretrained(args.model_path)
segment = 30 if args.large_datastore else 1 # Maximum number of segment: 144
data_files = []
for i in range(segment):
    if i>=100:
        data_files.append(f"data-00{i}-of-00144.parquet")
    elif i >=10:
        data_files.append(f"data-000{i}-of-00144.parquet")
    else:
        data_files.append(f"data-0000{i}-of-00144.parquet")
print("data_files:", data_files)

dataset = load_dataset('bigcode/the-stack-dedup', \
    data_dir='data/python', split='train', data_files=data_files)


datastore_path = './datastore_stack_large.idx' if args.large_datastore else './datastore_stack_small_codellama.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512 * 1024 * 1024,
    vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
)

print(tokenizer.vocab_size)

total_length = len(dataset)
print("number of samples: ", total_length)

# 加载两个tokenizer进行对比
# tokenizer_deepseek = AutoTokenizer.from_pretrained("/home/shape_model/deepseek-coder-6.7b-base")
# tokenizer_codellama = AutoTokenizer.from_pretrained("/home/shape_model/CodeLlama-7b-instruct-hf")

# 取一个样本进行对比
# sample_text = dataset.select(range(1))[0]['content']
# tokens_deepseek = tokenizer_deepseek.encode(sample_text)
# tokens_codellama = tokenizer_codellama.encode(sample_text)

# print(f"DeepSeek tokens length: {len(tokens_deepseek)}")
# print(f"CodeLlama tokens length: {len(tokens_codellama)}")

for sample in tqdm(dataset, total=len(dataset)):
    token_list = tokenizer.encode(sample['content'],truncation=True)
    # tokens_deepseek = tokenizer_deepseek.encode(sample['content'])
    # tokens_codellama = tokenizer_codellama.encode(sample['content'])

    # if len(tokens_deepseek)>10000 or len(tokens_codellama)>10000:
    #     print(f"DeepSeek tokens length: {len(tokens_deepseek)}")
    #     print(f"CodeLlama tokens length: {len(tokens_codellama)}")
    writer.add_entry(token_list)

writer.finalize()