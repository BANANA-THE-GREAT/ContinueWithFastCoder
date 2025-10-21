import threading
from flask import Flask, request, jsonify, Response
import os
import sys
sys.path.append("/home/jiaoziqian/LLMAcceleration-REST/")
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.my_utils import *
from rest.model.utils import *
import rest.model.my_utils
import draftretriever
from termcolor import colored

from tqdm import tqdm
import time
import argparse
import pandas as pd
import json
import heapq
from transformers import AutoTokenizer

app = Flask(__name__)

lock = threading.Lock()  # 线程锁，确保对全局变量的修改是线程安全的

stack_datastore = draftretriever.Reader(
    index_file_path='/home/jiaoziqian/LLMAcceleration-REST/datastore/datastore_stack_small_deepseek.idx',
)
empty_datastore = draftretriever.Reader(
    index_file_path='/home/jiaoziqian/LLMAcceleration-REST/dev_eval/empty.idx',
)
cache_sequences=[]

def generate_tokens_stream(model, tokenizer, max_token_span, num_draft, temperature, top_p, max_new_token, prompt, datastore, is_new_file):
    
    global service_interrupt
    global service_running
    global cache_sequences
    print(f"[useAcc]:{useAcc}")
    print(f"[isNewFile]:{is_new_file}")
    
    token_spans = list(range(2, max_token_span + 1))[::-1]

    #################
    skip_token=tokenizer.encode("\n", add_special_tokens=False)
    skip_token=skip_token[0]
    #################

    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    model.base_model.model.draft_mask = None
    logits = initialize_logits(
            input_ids, model, past_key_values
    )
    cur_length = input_len + 1
    new_token = 0

    if is_new_file:
        cache_sequences = []
    added_new_token=0  # 已经加入到缓存中的位置
    
    print("########")
    print(len(cache_sequences))
    print("########")

    #################
    repo_datastore = datastore
    if useAcc:
        datastores=[stack_datastore, repo_datastore]
    else:
        datastores=[]
    # print("datastore loaded!")
    #################

    

    with torch.inference_mode():
        for i in range(max_new_token):
            if useAcc:
                candidates, tree_candidates, draft_buffers, from_datastore = generate_candidates_and_draft_buffer_final(
                    skip_token,
                    global_miss_table,
                    logits,
                    input_ids,
                    datastores,
                    token_spans,
                    cache_sequences,
                    top_p,
                    temperature,
                    max_num_draft=num_draft,
                    device=model.base_model.device
                )
            else:
                candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                logits,
                input_ids,
                empty_datastore,
                token_spans,
                top_p,
                temperature,
                max_num_draft=num_draft,
                device=model.base_model.device
            )

            model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
            if not useAcc:
                candidates_type = "datastore"
            else:
                candidates_type = "datastore" if  from_datastore else "cache"

            logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    draft_buffers["draft_position_ids"],
                    input_ids,
                    draft_buffers["retrieve_indices"],
                )

            best_candidate, accept_length, best_candidate_original_sequence = evaluate_posterior_using_cache(
                    logits, candidates, temperature = temperature, top_p=top_p
                )

            source_type = candidates_type if best_candidate_original_sequence is not None else "model"
            
            if useAcc:
                # 更新缓存
                if best_candidate_original_sequence != None:
                    # 获取当前序列的实际长度
                    current_length = input_ids.shape[1]
                    # 取实际长度和max_token_span的最小值
                    span_length = min(current_length, max_token_span)
                    # 获取前文序列的最后span_length个token
                    current_sequence_ids = input_ids[0, -span_length:].to("cpu").tolist()  # 转换为CPU上的list
                    # 将检索序列和生成序列拼接并添加到缓存中
                    cache_sequence = current_sequence_ids + best_candidate_original_sequence.to("cpu").tolist()
                    cache_sequences.append(cache_sequence)
                
            input_ids, logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    draft_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )
            
            # Decode the new token(s)
            new_text = tokenizer.decode(input_ids[0, cur_length - 1:cur_length]).replace("\ufffd", "")
            cur_length += 1

            print(new_text, end="")
            # Yield the new token(s)
            yield json.dumps({"token": new_text, "source": source_type}) + "\n"
            # print("running", end=' ')
            print("[", end="")
            print(source_type, end="")
            print("]", end="")

            if useAcc:
                # 将模型生成的内容也加入到缓存中
                if new_token-added_new_token>20:
                    tokens_to_cache = input_ids[0, -(new_token-added_new_token):].to("cpu").tolist()  
                    cache_sequences.append(tokens_to_cache) 
                    added_new_token=new_token.item()  
                
            if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                break
            
            with lock:
                if service_interrupt:
                    print("interrupted")
                    service_interrupt = False
                    service_running = False
                    break
    with lock:
        service_interrupt = False
        service_running = False



def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/home/shared_models/workspace/deepseek-coder-6.7b-base",
        default="/home/shared_models/workspace/deepseek-coder-6.7b-base",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )

    # parser.add_argument(
    #     "--dataset-path",
    #     type=str,
    #     default="/home/zhaoqianhui/REST/dev_eval/data/dev_eval_trunc.jsonl",
    #     help="The path to the HumanEval dataset",
    # )

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )

    # # REST's hyperparameters
    # parser.add_argument(
    #     "--datastore-path",
    #     type=str,
    #     required=True,
    #     help="The path of the datastore for retrival.",
    # )

    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0
        
    print(args)

    model = RestModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    return args, model, tokenizer
   

# 执行预设代码
args, model, tokenizer = initialize()
# model_path="/home/shared_models/workspace/deepseek-coder-6.7b-base"
model_path="/home/shared_models/workspace/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
datastore_path = './datastore_customized_repo.idx'
writer = None
global_miss_table = set()

service_running = False
service_interrupt = False
datastore = draftretriever.Reader(
        index_file_path=datastore_path,
    )
useAcc = False
    
@contextmanager
def stream_response_context():
    try:
        yield  # 生成流式响应
    finally:
        # 在流式响应结束后执行
        # torch.cuda.empty_cache()  # 清理 GPU 显存
        with lock:
            global service_interrupt
            global service_running
            service_interrupt = False
            service_running = False

@app.route('/run_service', methods=['POST'])
def run_service():
    global service_running
    global service_interrupt
    
    with lock:
        if service_running:
            service_interrupt = True
            print("busy")
            return "busy", 200
        service_running = True
        
    data = request.get_json()
    # with open("req.json", 'w') as req:
    #     print(data, file=req)
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt parameter'}), 400
    
    global useAcc
    useAcc = data['useAcc']
    is_new_file = data['isNewFile']
    # useAcc = True if data['useAcc'] == "True" else False
    # is_new_file = True if data['isNewFile'] == 'True' else False
    

    def generate_stream():
        token_count = 0
        with stream_response_context():
            for token in generate_tokens_stream(
                model, tokenizer, args.max_token_span, args.num_draft,
                args.temperature, args.top_p, args.max_new_token, data['prompt'], datastore,
                is_new_file
            ):
                token_count += 1
                if token_count > 128:
                    with lock:
                        global service_interrupt
                        service_interrupt = True
                    break  # 超过64次停止生成
                yield token

    return Response(generate_stream(), content_type='application/json')

writer_key_keeped = False
@app.route('/init_writer', methods=['POST'])
def init_writer():
    global writer_key_keeped
    if writer_key_keeped:
        return "busy", 200
    writer_key_keeped = True
    global writer
    writer = draftretriever.Writer(
        index_file_path=datastore_path,
        max_chunk_len=512 * 1024 * 1024,
        vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
    )
    return "Writer initialized successfully!", 200

@app.route('/append_writer', methods=['POST'])
def append_writer():
    data = request.json
    content = data['content']
    # print(content)
    token_list = tokenizer.encode(content,truncation=True)
    writer.add_entry(token_list)
    return "Appended successfully!", 200

@app.route('/finalize_writer', methods=['POST'])
def finalize_writer():
    writer.finalize()
    global global_miss_table
    global_miss_table = set()
    global writer_key_keeped
    writer_key_keeped = False

    global datastore
    # Load datastore
    datastore = draftretriever.Reader(
        index_file_path=datastore_path,
    )
    return "Writer finalized successfully!", 200

@app.route('/init_cache', methods=['POST'])
def init_cache():
    global cache_sequences
    cache_sequences = []
    return "Cache initialized successfully!", 200

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=8000)