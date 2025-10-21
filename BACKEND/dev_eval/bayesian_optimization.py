from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import math
import random

import sys
import json
sys.path.append("../")
from termcolor import colored
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.utils import *
import draftretriever
from collections import deque

from tqdm import tqdm
import time
import argparse



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

class RetrievalSystem:
    def __init__(self, retrieval_datastores, init_satisfaction, beta, theta, T):
        self.retrieval_datastores = retrieval_datastores
        self.init_satisfaction = init_satisfaction
        self.current_datastore_index = 0
        self.satisfaction = init_satisfaction
        self.beta = beta
        self.theta = theta
        self.T = T
        
        # 历史表现相关
        self.history_window = deque(maxlen=5)  # 保存最近5次的接受长度
        
        # 动量相关
        self.momentum = 0
        self.momentum_factor = 0.9  # 动量因子
        
    def update_satisfaction(self, accept_length):
        # 1. 更新历史窗口
        self.history_window.append(accept_length)
        avg_length = sum(self.history_window) / len(self.history_window)
        
        # 2. 计算自适应学习率
        # 当满意度接近0或1时，更新步长变小
        adaptive_beta = self.beta * (1 - abs(2 * self.satisfaction - 1))
        
        # 3. 计算基于历史平均长度的delta
        base_delta = adaptive_beta * math.tanh(avg_length - 1)
        
        # 4. 更新动量
        self.momentum = self.momentum_factor * self.momentum + (1 - self.momentum_factor) * base_delta
        
        # 5. 应用更新
        self.satisfaction += self.momentum
        self.satisfaction = max(0, min(1, self.satisfaction))

    def should_switch(self):
        if self.satisfaction < self.theta:
            # 使用sigmoid函数使概率变化更平滑
            P = 1 / (1 + math.exp((self.satisfaction - self.theta) / self.T))
            return random.random() < P
        return False

    def next_datastore(self):
        self.current_datastore_index = (self.current_datastore_index + 1) % len(self.retrieval_datastores)
        self.satisfaction = self.init_satisfaction  # 重置满意度
        self.momentum = 0  # 重置动量
        self.history_window.clear()  # 清空历史窗口

    def update(self, accept_length):
        self.update_satisfaction(accept_length)
        if self.should_switch():
            self.next_datastore()

    def get_current_datastore(self):
        return self.retrieval_datastores[self.current_datastore_index]


def run_eval(model, tokenizer, dataset, init_satisfaction, beta, theta, T, max_token_span, num_draft, temperature, top_p, max_new_token):
    accept_lengths_tree_average = []
    avg_time_per_token_list = []

    accept_lengths_tree_average_micro = []
    avg_time_per_token_list_micro = []
    token_spans = list(range(2, max_token_span + 1))[::-1]
    print("token_spans: ", token_spans)

    for topic in dataset:
        for repo in dataset[topic]:
            repo_dataset=dataset[topic][repo]
            # print(colored(f'{topic}-{repo}',"blue"))

            # load datastore
            # print("loading the datastore ...")
            datastore_repo = draftretriever.Reader(
                # index_file_path='/home/zhaoqianhui/ADED/datastore/datastore_stack_large.idx',
                # index_file_path='/home/zhaoqianhui/ADED/datastore/datastore_stack_small.idx',
                # index_file_path='/home/zhaoqianhui/ADED/datastore/datastore_stack_mini.idx',
                index_file_path=f'/home/zhaoqianhui/REST/datastore/repo_datastore/{topic}/datastore_repo_{repo}.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/identifier_datastore/{namespace}.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/stack_repo_datastore/{topic}/datastore_stack_repo_{repo}.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/stackmini_repo_datastore/{topic}/datastore_stack_repo_{repo}.idx',
            )
            # load identifier datastore
            # print("loading the datastore ...")
            datastore_identifier = draftretriever.Reader(
                # index_file_path='/home/zhaoqianhui/ADED/datastore/datastore_stack_large.idx',
                # index_file_path='/home/zhaoqianhui/REST/datastore/datastore_stack_small.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/repo_datastore/{topic}/datastore_repo_{repo}.idx',
                index_file_path=f'/home/zhaoqianhui/REST/datastore/repo_identifier_datastore/{topic}/datastore_repo_identifier_{repo}.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/identifier_punctuation_datastore/{namespace}.idx',
                # index_file_path=f'/home/zhaoqianhui/ADED/datastore/stack_repo_datastore/{topic}/datastore_stack_repo_{repo}.idx',
            )
            # print("datastore loaded!")
            # print("datastore loaded!")

            for sample in repo_dataset:
                prompt = sample['contexts_above'] + sample['input_code']
                topic=sample['project_path'].split('/')[0]
                repo=sample['project_path'].split('/')[1]
                namespace=sample['namespace']



                retrieval_system = RetrievalSystem([datastore_repo, datastore_identifier], init_satisfaction, beta, theta, T)

                accept_lengths_tree = []
                with torch.inference_mode():

                    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
                    model.past_key_values = past_key_values
                    model.past_key_values_data = past_key_values_data
                    model.current_length_data = current_length_data

                    model.current_length_data.zero_() # this is for rerun


                    new_token = 0
                    # 增加truncation
                    max_length = model.config.max_position_embeddings//2
                    input_ids = tokenizer([prompt]).input_ids
                    # print(type(input_ids))
                    # print(type(input_ids[0]))
                    if len(input_ids[0]) > max_length:
                        input_ids[0] = input_ids[0][-max_length:]
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

                        datastore=retrieval_system.get_current_datastore()

                        # 只有这里调用了datastore，需要在在此之前决策是采用repo还是identifier
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
                                logits, candidates, temperature = temperature, top_p=top_p
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
                        
                        accept_length_tree = input_ids.shape[1] - cur_length
                        cur_length = accept_length_tree + cur_length
                        accept_lengths_tree.append(accept_length_tree)
                    
                        if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                            break

                        retrieval_system.update(accept_length_tree)

                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    avg_time_per_token = total_time / (new_token.cpu())
                    avg_time_per_token_list.append(avg_time_per_token)
                    avg_time_per_token_list_micro.append((total_time, new_token.cpu()))
                    
                    accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
                    accept_lengths_tree_average_micro.extend(accept_lengths_tree)

    print("accept_lengths_tree_average: ", np.mean(accept_lengths_tree_average))
    print("accept_lengths_tree_average_micro: ", np.mean(accept_lengths_tree_average_micro))
    print("avg_time_per_token: ", np.mean(avg_time_per_token_list))
    avg_time_per_token_micro = np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro])
    print("avg_time_per_token_micro: ", avg_time_per_token_micro)
    print("*"*30)
    print()
    return avg_time_per_token_micro


# 定义目标函数
def objective(init_satisfaction, beta, theta, T):
    model = RestModel.from_pretrained(
        "/home/shape_model/deepseek-coder-1.3b-base",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    dataset = load_dataset("/home/zhaoqianhui/ADED/dev_eval/LM_prompt_optimization.jsonl")

    avg_time_per_token_micro = run_eval(
                                            model, 
                                            tokenizer, 
                                            dataset,
                                            init_satisfaction,
                                            beta,
                                            theta,
                                            T,
                                            # datastore, 
                                            max_token_span=16,
                                            num_draft=64,
                                            temperature=0, 
                                            top_p=0,
                                            max_new_token=500
                                        )

    return -avg_time_per_token_micro


# pbounds = {
#     'init_satisfaction': (0, 1),
#     'beta': (0, 0.5),
#     'theta': (0.5, 1),
#     'T': (0,50)
# }

# 定义超参数
pbounds = {
    'init_satisfaction': (0.3, 0.7),   # 建议避免极端值
    'beta': (0.01, 0.2),              # 较小的学习率范围可能更稳定
    'theta': (0.2, 0.8),              # 切换阈值不宜过高或过低
    'T': (0.1, 10)                    # 温度参数范围可以适当缩小
}


optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)

# 记录优化过程
logger = JSONLogger(path="./logs1203.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# 最大化目标函数
# optimizer.maximize(init_points=12, n_iter=80)
optimizer.maximize(init_points=20, n_iter=200)


# 获取最优结果
max_point = optimizer.max
print(f"Maximum found at: {max_point}")