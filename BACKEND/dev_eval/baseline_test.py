import os
import sys
sys.path.append("../")
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.utils import *

from tqdm import tqdm
import time
import argparse
import json

from human_eval.dataset import HumanEvalDataset

# def load_dataset(file):
#     dataset={}
#     f=open(file,'r',encoding='utf-8')
#     for l in f.readlines():
#         data=json.loads(l)
#         topic=data['project_path'].split('/')[0]
#         repo=data['project_path'].split('/')[1]
#         if topic not in dataset.keys():
#             dataset[topic]={}
#         if repo not in dataset[topic].keys():
#             dataset[topic][repo]=[]
#         dataset[topic][repo].append(data)
#     return dataset

def run_eval(model, tokenizer,temperature, top_p, max_new_token):
    avg_time_per_token_list = []
    avg_time_per_token_list_micro = []

    # stop_sequence = "```\n"
    # stop_sequence_ids = tokenizer(stop_sequence, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze().tolist()

    # print('stop_sequence_ids')
    # print(stop_sequence_ids)

    for sample in tqdm(dataset, total=len(dataset)):
        prompt = sample['prompt']
        # prompt = sample['contexts_above'] + sample['input_code']
        with torch.inference_mode():
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

            model.current_length_data.zero_() # this is for rerun


            new_token = 0
            max_length = model.config.max_position_embeddings//2
            input_ids = tokenizer([prompt]).input_ids
            if len(input_ids[0]) > max_length:
                input_ids[0] = input_ids[0][-max_length:]
            input_len = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            model.base_model.model.draft_mask = None
            outputs = model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
            new_token = 0
            # logits = initialize_logits(
            #         input_ids, model, past_key_values
            # )
            # cur_length = input_len + 1
            # accept_lengths_tree.append(1)

            
            torch.cuda.synchronize()
            start_time = time.time()
            for i in range(2000):
            #     candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
            #             logits,
            #             input_ids,
            #             datastore,
            #             token_spans,
            #             top_p,
            #             temperature,
            #             max_num_draft=num_draft,
            #             device=model.base_model.device
            #         )
                
            #     model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]

            #     logits, outputs = tree_decoding(
            #             model,
            #             tree_candidates,
            #             past_key_values,
            #             draft_buffers["draft_position_ids"],
            #             input_ids,
            #             draft_buffers["retrieve_indices"],
            #         )

            #     best_candidate, accept_length = evaluate_posterior(
            #             logits, candidates, temperature = temperature, top_p=top_p
            #         )
            #     input_ids, logits, new_token = update_inference_inputs(
            #             input_ids,
            #             candidates,
            #             best_candidate,
            #             accept_length,
            #             draft_buffers["retrieve_indices"],
            #             outputs,
            #             logits,
            #             new_token,
            #             past_key_values_data,
            #             current_length_data,
            #         )
                if top_p > 0:
                    assert top_p < 1, "top_p should between 0.0 and 1"
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
                    filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                    input_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    input_id = input_id.view(input_id.shape[0], 1)
                else:
                    input_id = outputs.logits[:, -1:].argmax(dim=-1)

                outputs = model.base_model(input_id, use_cache=True, past_key_values = past_key_values)
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                new_token += 1
                # if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token or is_sublist(input_ids[0, input_len:].tolist(),stop_sequence_ids):
                if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                    break

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            avg_time_per_token = total_time / new_token
            avg_time_per_token_list.append(avg_time_per_token)
            avg_time_per_token_list_micro.append((total_time, new_token))
            

    print("avg_time_per_token: ", np.mean(avg_time_per_token_list))
    print("avg_time_per_token_micro: ", np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro]))
    print("*"*30)
    print()

# def is_sublist(A, B):
#     return any(B == A[i:i+len(B)] for i in range(len(A)-len(B)+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # default="codellama/CodeLlama-7b-instruct-hf",
        # default="/home/shape_model/deepseek-coder-1.3b-base",
        default="/home/shared_models/workspace/deepseek-coder-6.7b-base",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        # default="./HumanEval.jsonl.gz",
        # default="/home/zhaoqianhui/ADED/dev_eval/LM_prompt_retrieval.jsonl",
        # default="/home/zhaoqianhui/ADED/dev_eval/LM_prompt_retrieval_communications.jsonl",
        default="/home/jiaoziqian/LLMAcceleration-REST/dev_eval/dev_eval_trunc_communication.jsonl",
        # # for deepseekcoder-instruct
        # default="/home/zhaoqianhui/REST/dev_eval/data/without_context_instruct_prompt_communications.jsonl",
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

    # dataset = HumanEvalDataset(args.dataset_path)
    # dataset = load_dataset(args.dataset_path)
    dataset = HumanEvalDataset(args.dataset_path)

    
    run_eval(
        model, 
        tokenizer, 
        args.temperature, 
        args.top_p,
        args.max_new_token
    )