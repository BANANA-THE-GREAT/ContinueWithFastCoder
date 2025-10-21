from flask import Flask, request, jsonify, Response
import os
import sys
sys.path.append("/home/jiaoziqian/REST/")
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.utils import *
import draftretriever
from termcolor import colored
from tqdm import tqdm
import time
import argparse
import pandas as pd
import json
from transformers import AutoTokenizer

app = Flask(__name__)

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/home/shape_model/deepseek-coder-6.7b-base",
        default="/home/shape_model/CodeLlama-7b-instruct-hf",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/jiaoziqian/LLMAcceleration-REST/dev_eval/dev_eval_trunc_communication.jsonl",
        help="The path to the HumanEval dataset",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=500,
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
    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=4,
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
    print(f"Tokenizer unknown token: {tokenizer.unk_token}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    return args, model, tokenizer

def generate_tokens_stream(model, tokenizer, max_token_span, num_draft, temperature, top_p, max_new_token, prompt, datastore):
    """
    流式生成 token 的函数
    """
    token_spans = list(range(2, max_token_span + 1))[::-1]
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()

    # Initialize past key values
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    model.base_model.model.draft_mask = None
    logits = initialize_logits(input_ids, model, past_key_values)
    cur_length = input_len + 1
    new_token = 0

    with torch.inference_mode():
        for _ in range(max_new_token):
            candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                logits,
                input_ids,
                datastore,
                token_spans,
                top_p,
                temperature,
                max_num_draft=num_draft,
                device=model.base_model.device
            )

            model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]

            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                draft_buffers["draft_position_ids"],
                input_ids,
                draft_buffers["retrieve_indices"],
            )

            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature=temperature, top_p=top_p
            )

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
            new_text = tokenizer.decode(
                input_ids[0, cur_length - 1:cur_length], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )

            print(f"Generated token: {input_ids[0, cur_length - 1:cur_length]}, Decoded text: {new_text}")
            cur_length += 1

            # Yield the new token(s)
            yield json.dumps({"token": new_text}) + "\n"

            # Check for EOS token
            if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break



# 执行预设代码
args, model, tokenizer = initialize()
# model_path="/home/shape_model/deepseek-coder-6.7b-base"
model_path="/home/shape_model/CodeLlama-7b-instruct-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
datastore_path = './datastore_customized_repo.idx'
writer = None

@app.route('/run_service', methods=['POST'])
def run_service():
    data = request.json
    prompt = data['prompt']
    print(prompt)

    # Load datastore
    datastore = draftretriever.Reader(
        index_file_path=datastore_path,
    )

    # 返回流式响应
    return Response(
        generate_tokens_stream(
            model, tokenizer, args.max_token_span, args.num_draft, args.temperature, args.top_p, args.max_new_token, prompt, datastore
        ),
         content_type='application/json; charset=utf-8',  # 显式指定编码
    )

@app.route('/init_writer', methods=['POST'])
def init_writer():
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
    return "Writer finalized successfully!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)