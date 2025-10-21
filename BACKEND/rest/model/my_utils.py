import sys
sys.path.append("/home/jiaoziqian/REST/")
from rest.model.utils import *
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
from draftretriever import process_results
from typing import List
import random

def retrieve_from_cache(this_token, cache_sequences, choices=64):
    """
    基于输入token从缓存序列中检索并构建Trie树
    
    Args:
        this_token (list): 要匹配的token序列
        cache_sequences (list): 缓存的序列列表
        choices (int): 返回的最大候选序列数量
        
    Returns:
        tuple: (retrieved_token_list, draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices)
            - retrieved_token_list: 检索到的token序列列表
            - draft_attn_mask: 注意力掩码
            - tree_indices: 树索引
            - draft_position_ids: 位置编码
            - retrieve_indices: 检索索引
    """
    # Step 1: 检索匹配的序列
    matched_sequences = []
    
    # 遍历缓存序列
    for cache_seq in cache_sequences:
        # 将缓存序列转换为列表以便比较
        cache_tokens = cache_seq.tolist() if isinstance(cache_seq, torch.Tensor) else cache_seq
        
        # 在缓存序列中查找匹配的子序列
        for i in range(len(cache_tokens) - len(this_token) + 1):
            if cache_tokens[i:i+len(this_token)] == this_token:
                # 找到匹配后,获取后续的token序列
                next_tokens = cache_tokens[i+len(this_token):]
                if next_tokens:  # 确保有后续token
                    matched_sequences.append(next_tokens)
    
    # 如果没有找到匹配的序列，返回空结果
    if len(matched_sequences)==0:
        return [], [], [], [], []
    else:
        padded_paths, draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices = process_results(matched_sequences, choices)
    # retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
    
    # # Step 2: 构建Trie树
    # class TrieNode:
    #     def __init__(self):
    #         self.children = {}
    #         self.weight = 0  # 用于记录经过该节点的路径数量
    #         self.is_end = False
    
    # root = TrieNode()
    
    # # 将序列插入Trie树
    # for seq in matched_sequences:
    #     current = root
    #     for token in seq:
    #         if token not in current.children:
    #             current.children[token] = TrieNode()
    #         current = current.children[token]
    #         current.weight += 1
    #     current.is_end = True
    
    # # 剪枝
    # def prune_trie_by_weight(root, choices):
    #     """
    #     基于节点权重对Trie树进行剪枝,保留权重最大的choices个节点
    #     当遇到权重相同的节点时，优先保留层数较浅的节点
        
    #     Args:
    #         root: Trie树的根节点
    #         choices: 保留的最大节点数
    #     """
    #     # 初始化存储所有节点的列表
    #     nodes_info = []
    #     nodes_to_remove = set()
        
    #     def collect_nodes(node, prefix, depth):
    #         """收集所有节点及其前缀和深度"""
    #         for token, child in node.children.items():
    #             current_prefix = prefix + [token]
    #             # 将(权重,深度,前缀,节点)加入列表中
    #             nodes_info.append((child.weight, depth, current_prefix, child))
    #             collect_nodes(child, current_prefix, depth + 1)
        
    #     # 收集所有节点
    #     collect_nodes(root, [], 0)
        
    #     # 按权重升序、深度降序排序
    #     # 对于相同权重的节点，深度大的会排在前面（先被删除）
    #     nodes_info.sort(key=lambda x: (x[0], -x[1]))
        
    #     # 如果节点数超过choices,标记权重最小的节点待删除
    #     while len(nodes_info) > choices:
    #         weight, depth, prefix, node = nodes_info.pop(0)
    #         nodes_to_remove.add(tuple(prefix))
        
    #     def remove_marked_nodes(node, prefix=()):
    #         """移除被标记的节点"""
    #         to_remove = []
    #         for token, child in node.children.items():
    #             current_prefix = prefix + (token,)
    #             if current_prefix in nodes_to_remove:
    #                 to_remove.append(token)
    #             else:
    #                 remove_marked_nodes(child, current_prefix)
            
    #         # 删除标记的节点
    #         for token in to_remove:
    #             del node.children[token]
        
    #     # 执行删除操作
    #     remove_marked_nodes(root)

    #     # 计算每层的分支数
    #     def count_branches_by_level(node, level=0, level_counts=None):
    #         if level_counts is None:
    #             level_counts = {}
                
    #         # 统计当前节点的子节点数量
    #         branch_count = len(node.children)
            
    #         # 更新该层级的最大分支数
    #         level_counts[level] = max(level_counts.get(level, 0), branch_count)
            
    #         # 递归处理所有子节点
    #         for child in node.children.values():
    #             count_branches_by_level(child, level + 1, level_counts)
                
    #         return level_counts
        
    #     # 收集Trie树中所有路径
    #     def collect_paths(node, current_path=None):
    #         if current_path is None:
    #             current_path = []
                
    #         paths = []
            
    #         for token, child in node.children.items():
    #             new_path = current_path + [token]
    #             paths.append(new_path)
    #             # 递归收集子节点的路径
    #             paths.extend(collect_paths(child, new_path))
                
    #         return paths
        
       
    #     all_paths = collect_paths(root)
        
    #     # 计算最大路径长度
    #     max_length = max(len(path) for path in all_paths) if all_paths else 0

    #     def get_draft_choices(paths):
    #         path_dict = defaultdict(dict)
    #         cnt_dict = defaultdict(int)
    #         max_depth = max(len(path) for path in paths)

    #         # 初始化每层的计数器
    #         for depth in range(max_depth):
    #             cnt_dict[depth] = 0

    #         # 填充 path_dict 和 cnt_dict
    #         for path in paths:
    #             for depth, item in enumerate(path):
    #                 if item not in path_dict[depth]:
    #                     path_dict[depth][item] = cnt_dict[depth]
    #                     cnt_dict[depth] += 1

    #         # 计算最大分支数
    #         max_branch = max(len(v) for v in path_dict.values())

    #         # 生成草稿选择集
    #         draft_choices = set()
    #         for path in paths:
    #             for depth in range(len(path)):
    #                 draft_choice = [
    #                     path_dict[prev_depth][path[prev_depth]]
    #                     for prev_depth in range(depth + 1)
    #                 ]
    #                 draft_choices.add(tuple(draft_choice))

    #         # 转换为列表并返回
    #         draft_choices = [list(choice) for choice in draft_choices]
    #         return draft_choices, max_branch

    #     draft_choices, max_branch = get_draft_choices(all_paths)
        
    #     return all_paths, max_branch, draft_choices, max_length
    
    # # 剪枝Trie树并获取路径和最大长度
    # all_paths, max_branch, draft_choices, max_length = prune_trie_by_weight(root, choices)
    
    # # Step 3: 从剪枝后的Trie树生成draft buffers
    # draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices = generate_draft_buffers(draft_choices, max_branch)

    # padded_paths = [pad_path(path, max_length, -2) for path in all_paths]
    
    return padded_paths, draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices


def generate_draft_buffers(draft_choices, topk):
    """
    基于draft_choices生tree_attention
    
    Args:
        draft_choices (list): Trie树中的所有路径(注意存储的不是每个节点的值,而是每个节点在当前层的index,详细可参考知乎链接)
        topk (int): Trie树中每层分支数的最大值,用于后续计算偏移
        
    Returns:
        - draft_attn_mask: 注意力掩码
        - tree_indices: 树索引
        - draft_position_ids: 位置编码
        - retrieve_indices: 检索索引
    """

    # 根据长度和值对draft_choices进行排序
    sorted_draft_choices = sorted(draft_choices, key=lambda x: (len(x), x))
    
    draft_len = len(sorted_draft_choices) + 1
    assert draft_len <= 65, "draft_len should not exceed 65"
    
    # 统计每个深度的选择数量
    depth_counts = []
    prev_depth = 0
    for path in sorted_draft_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # 创建注意力掩码矩阵
    draft_attn_mask = [[0] * draft_len for _ in range(draft_len)]
    for i in range(draft_len):
        draft_attn_mask[i][0] = 1
        draft_attn_mask[i][i] = 1
    
    # 构建注意力掩码
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_draft_choice = sorted_draft_choices[start + j]
            if len(cur_draft_choice) == 1:
                continue
                
            ancestor_idx = []
            for c in range(len(cur_draft_choice) - 1):
                # 找到当前路径的所有祖先节点
                index = next(idx for idx, x in enumerate(sorted_draft_choices) 
                           if x[:min(c+1, len(x))] == cur_draft_choice[:min(c+1, len(cur_draft_choice))]) + 1
                ancestor_idx.append(index)
            
            for idx in ancestor_idx:
                draft_attn_mask[start + j + 1][idx] = 1
                
        start += depth_counts[i]
    
    # 生成树索引
    draft_tree_indices = [0] * draft_len
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_draft_choice = sorted_draft_choices[start + j]
            draft_tree_indices[start + j + 1] = cur_draft_choice[-1] + topk * i + 1
        start += depth_counts[i]
    
    # 生成位置编码
    draft_position_ids = [0] * draft_len
    start = 0
    for i in range(len(depth_counts)):
        for j in range(start + 1, start + depth_counts[i] + 1):
            draft_position_ids[j] = i + 1
        start += depth_counts[i]
    
    # 生成检索索引
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_draft_choices)):
        cur_draft_choice = sorted_draft_choices[-(i+1)]
        retrieve_indice = []
        if cur_draft_choice in retrieve_paths:
            continue
        else: 
            for c in range(len(cur_draft_choice)):
                retrieve_indice.append(sorted_draft_choices.index(cur_draft_choice[:c+1]))
                retrieve_paths.append(cur_draft_choice[:c+1])
            
        retrieve_indices_nest.append(retrieve_indice)
    
    # 填充检索索引
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    
    # 索引调整
    for i in range(len(retrieve_indices)):
        retrieve_indices[i] = [x + 1 for x in retrieve_indices[i]]
        retrieve_indices[i].insert(0, 0)
    
    return draft_attn_mask, draft_tree_indices, draft_position_ids, retrieve_indices


def generate_candidates_and_draft_buffer_using_cache(logits, input_ids, datastore, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    # 记录candidates是从cache还是datastore中得到的
    from_datastore = False

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    
    # search from cache
    if len(cache_sequences)>=50:
        # print('search from cache')
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()

            # retrieve from cache
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = retrieve_from_cache(
                this_token, 
                cache_sequences, 
                # choices=max_num_draft
                # choices=16
            )
            
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

    # search from datastore
    if len(retrieved_token_list) == 0:
        # print('search from datastore')
    # else:
        from_datastore = True
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)

            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break
    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers, from_datastore


def evaluate_posterior_using_cache(
    logits, candidates, temperature, top_p=0.8
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - top_p (float): Top-p sampling threshold.
    
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - original_sequence (torch.Tensor): The complete token sequence of the best candidate.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            original_sequence = None
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
             # 获取最佳候选项的完整序列
            original_sequence = candidates[best_candidate]
            
       
        
        return best_candidate, accept_length, original_sequence
        
    elif top_p > 0:
        assert top_p < 1.0, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            original_sequence = None
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # 获取最佳候选项的完整序列
            original_sequence = candidates[best_candidate]
        
        return best_candidate, accept_length, original_sequence
    else:
        raise NotImplementedError
    
# ========================================================
# for parallel search in cache and datastore
# lib.rs needs modification: return only the retrieved sequences
# ========================================================
def generate_candidates_and_draft_buffer_using_cache_parallel(logits, input_ids, datastore, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    # 记录candidates是从cache还是datastore中得到的
    # from_datastore = False

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    # parallel search from cache and datastore
    search_results=[]
    cache_results, datastore_results = cache_parallel_search(input_ids_extend, token_spans, cache_sequences, datastore, max_num_draft)

    # print(type(cache_results))
    # print(type(datastore_results))
    # print(type(search_results))
    # TODO: 可以给二者赋予不同的权重
    # print(f'cache results: {cache_results}')
    # print(f'datastore results: {datastore_results}')
    search_results.extend(cache_results)
    search_results.extend(datastore_results)
    # print(f'search results: {search_results}')
   

    # TODO: just continue to the next retrieval process
    if len(search_results) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        # cut to choices and generate tree attention
        # print(search_results)
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)
        
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers



def search_cache(input_ids_extend, token_spans, cache_sequences, max_num_draft):
    def search(this_token, cache_sequences):
        matched_sequences = []
        for cache_seq in cache_sequences:
            cache_tokens = cache_seq.tolist() if isinstance(cache_seq, torch.Tensor) else cache_seq
            
            for i in range(len(cache_tokens) - len(this_token) + 1):
                if cache_tokens[i:i+len(this_token)] == this_token:
                    next_tokens = cache_tokens[i+len(this_token):]
                    if next_tokens:  
                        matched_sequences.append(next_tokens)
        return matched_sequences
    
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
        cache_retrieved_token_list = search(this_token, cache_sequences)
        if len(cache_retrieved_token_list) > 0:
            return cache_retrieved_token_list
    return []

def search_datastore(input_ids_extend, token_spans, datastore, max_num_draft):
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
        retrieved_token_list = datastore.search_for_sequences(this_token, choices=max_num_draft)
        if len(retrieved_token_list) > 0:
            return retrieved_token_list
    return []

def cache_parallel_search(input_ids_extend, token_spans, cache_sequences, datastore, max_num_draft):
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交两个任务
        future_cache = executor.submit(
            search_cache, 
            input_ids_extend, 
            token_spans, 
            cache_sequences, 
            max_num_draft
        )
        future_datastore = executor.submit(
            search_datastore, 
            input_ids_extend, 
            token_spans, 
            datastore, 
            max_num_draft
        )
        
        # 获取结果
        cache_results = future_cache.result()
        datastore_results = future_datastore.result()
        
        return cache_results, datastore_results
    

def cut_to_choices_and_generate_attention(matched_sequences, choices=64):   
    # Step 2: 构建Trie树
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.weight = 0  # 用于记录经过该节点的路径数量
            self.is_end = False
    
    root = TrieNode()
    
    # 将序列插入Trie树
    for seq in matched_sequences:
        current = root
        for token in seq:
            if token not in current.children:
                current.children[token] = TrieNode()
            current = current.children[token]
            current.weight += 1
        current.is_end = True
    
    # 剪枝
    def prune_trie_by_weight(root, choices):
        # 初始化存储所有节点的列表
        nodes_info = []
        nodes_to_remove = set()
        
        def collect_nodes(node, prefix, depth):
            """收集所有节点及其前缀和深度"""
            for token, child in node.children.items():
                current_prefix = prefix + [token]
                # 将(权重,深度,前缀,节点)加入列表中
                nodes_info.append((child.weight, depth, current_prefix, child))
                collect_nodes(child, current_prefix, depth + 1)
        
        # 收集所有节点
        collect_nodes(root, [], 0)
        
        # 按权重升序、深度降序排序
        # 对于相同权重的节点，深度大的会排在前面（先被删除）
        nodes_info.sort(key=lambda x: (x[0], -x[1]))
        
        # 如果节点数超过choices,标记权重最小的节点待删除
        while len(nodes_info) > choices:
            weight, depth, prefix, node = nodes_info.pop(0)
            nodes_to_remove.add(tuple(prefix))
        
        def remove_marked_nodes(node, prefix=()):
            """移除被标记的节点"""
            to_remove = []
            for token, child in node.children.items():
                current_prefix = prefix + (token,)
                if current_prefix in nodes_to_remove:
                    to_remove.append(token)
                else:
                    remove_marked_nodes(child, current_prefix)
            
            # 删除标记的节点
            for token in to_remove:
                del node.children[token]
        
        # 执行删除操作
        remove_marked_nodes(root)

        # 计算每层的分支数
        def count_branches_by_level(node, level=0, level_counts=None):
            if level_counts is None:
                level_counts = {}
                
            # 统计当前节点的子节点数量
            branch_count = len(node.children)
            
            # 更新该层级的最大分支数
            level_counts[level] = max(level_counts.get(level, 0), branch_count)
            
            # 递归处理所有子节点
            for child in node.children.values():
                count_branches_by_level(child, level + 1, level_counts)
                
            return level_counts
        
        # 收集Trie树中所有路径
        def collect_paths(node, current_path=None):
            if current_path is None:
                current_path = []
                
            paths = []
            
            for token, child in node.children.items():
                new_path = current_path + [token]
                paths.append(new_path)
                # 递归收集子节点的路径
                paths.extend(collect_paths(child, new_path))
                
            return paths
        
       
        all_paths = collect_paths(root)
        
        # 计算最大路径长度
        max_length = max(len(path) for path in all_paths) if all_paths else 0

        def get_draft_choices(paths):
            path_dict = defaultdict(dict)
            cnt_dict = defaultdict(int)
            max_depth = max(len(path) for path in paths)

            # 初始化每层的计数器
            for depth in range(max_depth):
                cnt_dict[depth] = 0

            # 填充 path_dict 和 cnt_dict
            for path in paths:
                for depth, item in enumerate(path):
                    if item not in path_dict[depth]:
                        path_dict[depth][item] = cnt_dict[depth]
                        cnt_dict[depth] += 1

            # 计算最大分支数
            max_branch = max(len(v) for v in path_dict.values())

            # 生成草稿选择集
            draft_choices = set()
            for path in paths:
                for depth in range(len(path)):
                    draft_choice = [
                        path_dict[prev_depth][path[prev_depth]]
                        for prev_depth in range(depth + 1)
                    ]
                    draft_choices.add(tuple(draft_choice))

            # 转换为列表并返回
            draft_choices = [list(choice) for choice in draft_choices]
            return draft_choices, max_branch

        draft_choices, max_branch = get_draft_choices(all_paths)
        
        return all_paths, max_branch, draft_choices, max_length
    
    # 剪枝Trie树并获取路径和最大长度
    all_paths, max_branch, draft_choices, max_length = prune_trie_by_weight(root, choices)
    
    # Step 3: 从剪枝后的Trie树生成draft buffers
    draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices = generate_draft_buffers(draft_choices, max_branch)

    padded_paths = [pad_path(path, max_length, -2) for path in all_paths]
    
    return padded_paths, draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices



# ========================================================
# for parallel search in different datastores
# ========================================================
def generate_candidates_and_draft_buffer_for_datastores_parallel(weights, logits, input_ids, datastores, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    # parallel search from datastores
    search_results = datastores_parallel_search(weights, input_ids_extend, token_spans, datastores, max_num_draft)
   
    # TODO: just continue to the next retrieval process
    if len(search_results) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        # cut to choices and generate tree attention
        # print(search_results)
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)
        
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


def datastores_parallel_search(weights, input_ids_extend, token_spans, datastores, max_num_draft):
    with ThreadPoolExecutor(max_workers=len(datastores)) as executor:
        # 提交两个任务
        futures = [executor.submit(
            search_datastore, 
            input_ids_extend, 
            token_spans, 
            datastore, 
            max_num_draft
        ) for datastore in datastores]
        
        # 获取结果
        results = []
        # weights = [1, 2, 1]  # 每个datastore的权重，长度需要与datastores数量相同
        # print(len(datastores))
        # print(len(weights))
        # print(len(futures))
        assert len(weights) == len(futures)
        
        for weight, future in zip(weights, futures):
            result = future.result()
            for item in result:
                results.extend([item] * weight)  # 根据权重增加次数
        
        return results



# ========================================================
# for parallel search in different datastores and using cache
# parallel + cache
# ========================================================
def generate_candidates_and_draft_buffer_paralle_and_cache(logits, input_ids, datastores, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    # 记录candidates是从cache还是datastore中得到的
    from_datastore = False

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    
    # search from cache
    if len(cache_sequences)>=50:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()

            # retrieve from cache
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = retrieve_from_cache(
                this_token, 
                cache_sequences, 
                # choices=max_num_draft
                # choices=16
            )
            
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

    # parallel search from datastores
    if len(retrieved_token_list) == 0:
    # else:
        from_datastore = True
        search_results = datastores_parallel_search([1,1],input_ids_extend, token_spans, datastores, max_num_draft)
        if len(search_results)>0:
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)


    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers, from_datastore


# ========================================================
# for time selection
# not retrieve after whitespace
# ========================================================
def generate_candidates_and_draft_buffer_time(skip_token, logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
   

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)

    
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
     
    # 判断刚才生成的token是否为空白符，若是则检索
    if input_ids_extend[-1].item() != skip_token:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break
    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers

# ========================================================
# for time selection
# not retrieve after whitespace [with probability]
# ========================================================
def generate_candidates_and_draft_buffer_time_probability(probability, skip_token, logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
   

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)

    
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
     
    # 判断刚才生成的token是否为空白符，若是则以一定的概率不检索
    # 如果一定不检索 那就是probability=1的情况
    should_retrieve = True
    if input_ids_extend[-1].item() == skip_token and random.random() < probability:
        should_retrieve = False
        
    if should_retrieve:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break
    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


# ========================================================
# for time selection
# miss table
# ========================================================
# miss = 0
# retrieval = 0
# not_retrieval = 0
# total = 0
def generate_candidates_and_draft_buffer_miss(global_miss_table, logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    global miss
    global retrieval
    global not_retrieval
    global total
    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
    miss_token = input_ids_extend.squeeze(0)[-2:].to("cpu").tolist()
    is_miss = False
    # 总次数
    total = total + 1
    if tuple(miss_token) in global_miss_table:
        # miss 的次数
        miss = miss + 1
        is_miss = True
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
    if not is_miss:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break
    # 上面token_span从16逐步下降到2, 如果到2都还没有检索到，就说明彻底检索不到了
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0 or is_miss:
        # Just randomlt guess one token
        if not is_miss:
            # 没检索到的次数
            not_retrieval = not_retrieval + 1
            global_miss_table.add(tuple(miss_token))
        
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        # 检索到的次数
        retrieval = retrieval + 1
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


# ========================================================
# for time selection
# miss table + not retrieve
# ========================================================
# miss = 0
# retrieval = 0
# not_retrieval = 0
# total = 0
def generate_candidates_and_draft_buffer_time_selection_all(skip_token, global_miss_table, logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    global miss
    global retrieval
    global not_retrieval
    global total
    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)

    miss_token = input_ids_extend.squeeze(0)[-2:].to("cpu").tolist()
    is_miss = False
    # 总次数
    total = total + 1
    if tuple(miss_token) in global_miss_table:
        # miss 的次数
        miss = miss + 1
        is_miss = True

    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
    if (not is_miss) and input_ids_extend[-1].item() != skip_token:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break
    # 上面token_span从16逐步下降到2, 如果到2都还没有检索到，就说明彻底检索不到了
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0 or is_miss:
        # Just randomlt guess one token
        if not is_miss:
            # 没检索到的次数
            not_retrieval = not_retrieval + 1
            global_miss_table.add(tuple(miss_token))
        
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        # 检索到的次数
        retrieval = retrieval + 1
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers



# ========================================================
# final strategy
# cache + parallel + time selection
# ========================================================
miss = 0
retrieval = 0
not_retrieval = 0
total = 0
def generate_candidates_and_draft_buffer_final(skip_token, global_miss_table, logits, input_ids, datastores, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    # 记录candidates是从cache还是datastore中得到的
    from_datastore = False

    global miss
    global retrieval
    global not_retrieval
    global total

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    miss_token = input_ids_extend.squeeze(0)[-2:].to("cpu").tolist()
    is_miss = False
    # 总次数
    total = total + 1
    if tuple(miss_token) in global_miss_table:
        # miss 的次数
        miss = miss + 1
        is_miss = True

    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []


    # search from cache
    if len(cache_sequences)>=50:
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()

            # retrieve from cache
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = retrieve_from_cache(
                this_token, 
                cache_sequences, 
                # choices=max_num_draft
                # choices=16
            )
            
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

    
    # # parallel search from datastores
    # if len(retrieved_token_list) == 0:
    #     if (not is_miss) and input_ids_extend[-1].item() != skip_token:
    #         from_datastore = True
    #         search_results = datastores_parallel_search([1,1],input_ids_extend, token_spans, datastores, max_num_draft)
    #         if len(search_results)>0:
    #          retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)

    # parallel search from datastores probability=0.2
    if len(retrieved_token_list) == 0:
        should_retrieve = True
        if is_miss or (input_ids_extend[-1].item() == skip_token and random.random() < 0.5):
            should_retrieve = False
        if should_retrieve:
            from_datastore = True
            search_results = datastores_parallel_search([1,1],input_ids_extend, token_spans, datastores, max_num_draft)
            if len(search_results)>0:
                retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)


    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        if not is_miss:
            # 没检索到的次数
            not_retrieval = not_retrieval + 1
            global_miss_table.add(tuple(miss_token))

        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers, from_datastore


# ========================================================
# parallel + time selection
# ========================================================
def generate_candidates_and_draft_buffer_parallel_and_time(skip_token, global_miss_table, weights, logits, input_ids, datastores, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    global miss
    global retrieval
    global not_retrieval
    global total
    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)

    miss_token = input_ids_extend.squeeze(0)[-2:].to("cpu").tolist()
    is_miss = False
    # 总次数
    total = total + 1
    if tuple(miss_token) in global_miss_table:
        # miss 的次数
        miss = miss + 1
        is_miss = True

        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    # if (not is_miss) and input_ids_extend[-1].item() != skip_token:
    # # parallel search from datastores
    #     search_results = datastores_parallel_search(weights, input_ids_extend, token_spans, datastores, max_num_draft)

    should_retrieve = True
    if is_miss or (input_ids_extend[-1].item() == skip_token and random.random() < 0.5):
        should_retrieve = False
    if should_retrieve:
    # parallel search from datastores
        search_results = datastores_parallel_search(weights, input_ids_extend, token_spans, datastores, max_num_draft)


    # TODO: just continue to the next retrieval process
    if is_miss or len(search_results) == 0:
        # Just randomlt guess one token
        if not is_miss:
            # 没检索到的次数
            not_retrieval = not_retrieval + 1
            global_miss_table.add(tuple(miss_token))
        
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        # cut to choices and generate tree attention
        # print(search_results)
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = process_results(search_results, choices=max_num_draft)
        
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


# ========================================================
# cache + time selection
# ========================================================
def generate_candidates_and_draft_buffer_cache_and_time(skip_token, global_miss_table, logits, input_ids, datastore, token_spans, cache_sequences, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    # 记录candidates是从cache还是datastore中得到的
    from_datastore = False

    global miss
    global retrieval
    global not_retrieval
    global total

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    miss_token = input_ids_extend.squeeze(0)[-2:].to("cpu").tolist()
    is_miss = False
    # 总次数
    total = total + 1
    if tuple(miss_token) in global_miss_table:
        # miss 的次数
        miss = miss + 1
        is_miss = True
    
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []

    
    # search from cache
    if len(cache_sequences)>=50:
        # print('search from cache')
        for span_id, token_span in enumerate(token_spans):
            this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()

            # retrieve from cache
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = retrieve_from_cache(
                this_token, 
                cache_sequences, 
                # choices=max_num_draft
                # choices=16
            )
            
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

    # search from datastore
    if len(retrieved_token_list) == 0:
        # print('search from datastore')
    # else:
        # from_datastore = True
        should_retrieve = True
        if is_miss or (input_ids_extend[-1].item() == skip_token and random.random() < 0.5):
            should_retrieve = False
            
        if should_retrieve:
            from_datastore = True
            for span_id, token_span in enumerate(token_spans):
                this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
                # Retrieve draft tokens from the datastore, and get draft buffer
                retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)

                # No retrieved sequences
                if len(retrieved_token_list) == 0:
                    continue
                # Break because this span has hitted
                else:
                    break
    
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers, from_datastore