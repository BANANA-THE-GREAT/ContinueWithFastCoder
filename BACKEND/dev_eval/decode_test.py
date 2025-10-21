import os
import sys
# sys.path.append("../")
sys.path.append("/home/jiaoziqian/LLMAcceleration-REST/")
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
# from rest.model.utils import *
from rest.model.my_utils import *
import rest.model.my_utils
import draftretriever
from termcolor import colored

from tqdm import tqdm
import time
import argparse
import pandas as pd
import json
import heapq


def load_dataset(file):
    dataset={}
    f=open(file,'r',encoding='utf-8')
    for l in f.readlines():
        data=json.loads(l)
        topic=data['project_path'].split('/')[0]
        repo=data['project_path'].split('/')[1]
        if topic not in dataset.keys():
            dataset[topic]={}
        if repo not in dataset[topic].keys():
            dataset[topic][repo]=[]
        dataset[topic][repo].append(data)
    return dataset

def run_eval(model, tokenizer, max_token_span, num_draft, temperature, top_p, max_new_token):
    fw=open('0116.txt','a',encoding='utf-8')
    # fw.write('[final][codellama-7b]\n')
    accept_lengths_tree_average = []
    avg_time_per_token_list = []

    accept_lengths_tree_average_micro = []
    avg_time_per_token_list_micro = []
    # accept_length_datastore=[]
    # accept_length_cache=[]
    token_spans = list(range(2, max_token_span + 1))[::-1]
    print("token_spans: ", token_spans)

    print('skip_token:')
    skip_token=tokenizer.encode("\n", add_special_tokens=False)
    print(skip_token)
    skip_token=skip_token[0]
    print(skip_token) # 32013

    # fp=open(f'/home/zhaoqianhui/REST/dev_eval/retrieved_tokens/compare_cache_and_datastore_new.jsonl','a',encoding='utf-8')

    stack_datastore = draftretriever.Reader(
                # index_file_path='/home/zhaoqianhui/REST/datastore/datastore_stack_small_deepseek.idx',
                index_file_path='/home/zhaoqianhui/REST/datastore/datastore_stack_small_deepseek.idx',
            )

    for topic in dataset:
        # if topic == 'Communications':
        for repo in dataset[topic]:
            repo_dataset=dataset[topic][repo]
            print(colored(f'{topic}-{repo}',"blue"))
            if topic == "Scientific-Engineering" and repo == "folium":
                continue
            # load datastore
            # print("loading the datastore ...")
            repo_datastore = draftretriever.Reader(
                # index_file_path=f'/home/zhaoqianhui/REST/datastore/repo_datastore_cl/{topic}/datastore_repo_{repo}.idx',
                index_file_path=f'/home/zhaoqianhui/REST/datastore/deveval/repoandngram_datastore/{topic}/datastore_repoandngram_{repo}.idx',
            )
            # print("datastore loaded!")
            datastores=[stack_datastore, repo_datastore]

            miss_before = rest.model.my_utils.miss  
            total_before = rest.model.my_utils.total
            retrieval_before = rest.model.my_utils.retrieval
            not_retrieval_before = rest.model.my_utils.not_retrieval
            # miss table
            global_miss_table = set()

            for sample in tqdm(repo_dataset, total=len(repo_dataset)):
                prompt = sample['prompt']
                # prompt = sample['contexts_above'] + sample['input_code']
                # prompt = sample['input_code']
                topic=sample['project_path'].split('/')[0]
                repo=sample['project_path'].split('/')[1]
                namespace=sample['namespace']
                cache_sequences = []

                accept_lengths_tree = []
                added_new_token=0  # 已经加入到缓存中的位置
                # # 添加记录token和当前生成文本
                # cache_retrieved_tokens = []
                # datastore_retrieved_tokens = []
                # current_text = ''
                with torch.inference_mode():

                    # Initialize the past key and value states
                    if hasattr(model, "past_key_values"):
                        past_key_values = model.past_key_values
                        past_key_values_data = model.past_key_values_data
                        current_length_data = model.current_length_data
                        # Reset the past key and value states
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


                    new_token = 0
                    # 增加truncation
                    # max_length = model.config.max_position_embeddings//2
                    # max_length = 1024
                    input_ids = tokenizer([prompt]).input_ids
                    # print(type(input_ids))
                    # print(type(input_ids[0]))
                    # if len(input_ids[0]) > max_length:
                    #     input_ids[0] = input_ids[0][-max_length:]
                    input_len = len(input_ids[0])
                    input_ids = torch.as_tensor(input_ids).cuda()

                    model.base_model.model.draft_mask = None
                    logits = initialize_logits(
                            input_ids, model, past_key_values
                    )
                    cur_length = input_len + 1
                    accept_lengths_tree.append(1)
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for i in range(2000):

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
                        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]

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
                        
                        accept_length_tree = input_ids.shape[1] - cur_length
                        cur_length = accept_length_tree + cur_length
                        accept_lengths_tree.append(accept_length_tree)

                        # if accept_length_tree > 1:
                        #     # accept_tokens = tokenizer.decode(candidates[0, :accept_length])
                        #     # print(accept_tokens)
                        #     start_idx = len(current_text)
                        #     tmp = current_text
                        #     current_text = tokenizer.decode(input_ids[0, input_len:])
                        #     # current_text += accept_tokens
                        #     end_idx = len(current_text)
                        #     accept_tokens = current_text[len(tmp):]

                        #     item = {
                        #         'tokens': accept_tokens,
                        #         'start_idx': start_idx,
                        #         'end_idx': end_idx,
                        #         'accept_token_length': accept_length_tree
                        #     }

                        #     if from_datastore: 
                        #         datastore_retrieved_tokens.append(item)
                        #     else:
                        #         cache_retrieved_tokens.append(item)
                        # else:
                        #     current_text = tokenizer.decode(input_ids[0, input_len:])

                        # 将模型生成的内容也加入到缓存中
                        if new_token-added_new_token>20:
                            tokens_to_cache = input_ids[0, -(new_token-added_new_token):].to("cpu").tolist()  
                            cache_sequences.append(tokens_to_cache)
                            # print('-----------------------------------------------------------------------')
                            # print(colored("Decoded tokens to cache: " + tokenizer.decode(input_ids[0, -(new_token-added_new_token):]), 'green'))  
                            added_new_token=new_token.item()  
                            


                        if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                            # full_generated_text=tokenizer.decode(input_ids[0, input_len:])
                            # print(len(cache_sequences))
                            # print(cache_sequences)
                            break

                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    avg_time_per_token = total_time / (new_token.cpu())
                    avg_time_per_token_list.append(avg_time_per_token)
                    avg_time_per_token_list_micro.append((total_time, new_token.cpu()))
                    
                    accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
                    accept_lengths_tree_average_micro.extend(accept_lengths_tree)

                #     if from_datastore:
                #         accept_length_datastore.extend(accept_lengths_tree)
                #     else:
                #         accept_length_cache.extend(accept_lengths_tree)


                # json.dump({
                #     'project_path': sample['project_path'],
                #     'namespace': namespace,
                #     'full_generated_text': full_generated_text,
                #     'cache_retrieved_tokens': cache_retrieved_tokens,
                #     'datastore_retrieved_tokens': datastore_retrieved_tokens,
                # }, fp, ensure_ascii=False)
                # fp.write('\n')
            repo_miss = rest.model.my_utils.miss - miss_before
            repo_total = rest.model.my_utils.total - total_before
            repo_retrieval = rest.model.my_utils.retrieval - retrieval_before
            repo_not_retrieval = rest.model.my_utils.not_retrieval - not_retrieval_before
            miss_result = {
                "topic": topic,
                "repo": repo,
                "total": repo_total,
                "miss": repo_miss,
                "retrieval": repo_retrieval,
                "not_retrieval": repo_not_retrieval,
                "miss_table_len": len(global_miss_table)
            }
            # with open("result/1_3b_repo.jsonl", "a") as f:
            #     f.write(json.dumps(miss_result) + "\n")



    print("accept_lengths_tree_average: ", np.mean(accept_lengths_tree_average))
    fw.write(f"accept_lengths_tree_average: {np.mean(accept_lengths_tree_average)}\n")
    
    print("accept_lengths_tree_average_micro: ", np.mean(accept_lengths_tree_average_micro))
    fw.write(f"accept_lengths_tree_average_micro: {np.mean(accept_lengths_tree_average_micro)}\n")
    
    # print("accept_lengths_cache: ", np.mean(accept_length_cache))
    # print("accept_lengths_datastore: ", np.mean(accept_length_datastore))
    
    print("avg_time_per_token: ", np.mean(avg_time_per_token_list))
    fw.write(f"avg_time_per_token: {np.mean(avg_time_per_token_list)}\n")
    
    avg_time_micro = np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro])
    print("avg_time_per_token_micro: ", avg_time_micro)
    fw.write(f"avg_time_per_token_micro: {avg_time_micro}\n")
    
    print("*"*30)
    fw.write("*"*30 + "\n")
    print()
    fw.write("\n")
    
    fw.flush()  # 确保数据及时写入文件
    # print("accept_lengths_tree_average: ", np.mean(accept_lengths_tree_average))
    # print("accept_lengths_tree_average_micro: ", np.mean(accept_lengths_tree_average_micro))
    # print("avg_time_per_token: ", np.mean(avg_time_per_token_list))
    # print("avg_time_per_token_micro: ", np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro]))
    print("total: ", rest.model.my_utils.total)
    print("miss: ", rest.model.my_utils.miss)
    print("miss / total(100%): ", 100 * rest.model.my_utils.miss / rest.model.my_utils.total)
    print("retrieval / total(100%): ", 100 * rest.model.my_utils.retrieval / rest.model.my_utils.total)
    print("not_retrieval / total(100%): ", 100 * rest.model.my_utils.not_retrieval / rest.model.my_utils.total)
    print("*"*30)
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/home/zhaoqianhui/workspace/model/CodeLlama-7b-Python-hf",
        # default="/home/shape_model/deepseek-coder-1.3b-base",
        default="/home/shape_model/deepseek-coder-6.7b-base",
        # default="/home/shape_model/Qwen2.5-Coder-1.5B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        # default="../human_eval/HumanEval.jsonl.gz",
        # default="/home/zhaoqianhui/ADED/dev_eval/LM_prompt_retrieval.jsonl",
        # default="/home/zhaoqianhui/ADED/dev_eval/LM_prompt_retrieval_communications.jsonl",
        # default="/home/zhaoqianhui/REST/dev_eval/data/tiaoshi.jsonl",
        default="/home/zhaoqianhui/REST/dev_eval/data/dev_eval_trunc.jsonl",
        # default="/home/zhaoqianhui/REST/dev_eval/data/dev_eval_trunc_communications.jsonl",
        help="The path to the HumanEval dataset",
    )
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

    # dataset = HumanEvalDataset(args.dataset_path)
    # df = pd.read_json('/home/zhaoqianhui/DevEval/my_data/LM_prompt_elements.jsonl', lines=True)
    # dataset = df.to_dict(orient='records')
    # dataset=[]
    # f=open("/home/zhaoqianhui/ADED/dev_eval/LM_prompt_retrieval_communications.jsonl",'r',encoding='utf-8')
    # for l in f.readlines():
    #     dataset.append(json.loads(l))

    dataset = load_dataset(args.dataset_path)

    # print("loading the datastore ...")
    # datastore = draftretriever.Reader(
    #             index_file_path=args.datastore_path,
    #         )
    # print("datastore loaded!")
    
    run_eval(
        model, 
        tokenizer, 
        # datastore, 
        args.max_token_span,
        args.num_draft,
        args.temperature, 
        args.top_p,
        args.max_new_token
    )