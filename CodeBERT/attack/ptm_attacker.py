import sys
import os
import time

sys.path.append('../dataset/')
sys.path.append('../code/')   # CodeBERT, UniXcoder, GraphCodeBERT, EPVD, ReGVD
sys.path.append('../../dataset/python_parser')
from run import convert_examples_to_features, TextDataset, InputFeatures, convert_code_to_features
from run_parser import get_identifiers_with_tokens
from data_augmentation import get_identifiers, get_example, get_code_tokens
import csv
import subprocess
import copy
import re
import json
import logging
import argparse
import warnings
from data_utils import is_valid_identifier
from attack_utils import get_masked_code_by_position, _tokenize, insert_at_line
from attack_utils import map_chromesome, EPVDDataset, CodeDataset, GraphCodeDataset, get_identifier_posistions_from_code
import torch
import numpy as np
import random
from model import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed
from python_parser.run_parser import  get_example, get_example_pos, get_identifiers_list, \
    is_valid_variable_c, get_all_snippet, remove_comments_and_docstrings, get_var, add_dec, get_snippet_token,get_identifiers_coda,get_example_batch,get_code_style, change_code_style,get_example_coda
from python_parser.run_parser import mutation_tra

from scipy.spatial.distance import cosine as cosine_distance
from python_parser.run_parser import mutation
# 加载英文模型
nlp = spacy.load('en_core_web_md')


# 定义一个函数来获取词向量的均值
def get_mean_vector(group):
    vectors = [nlp(word).vector for word in group if nlp(word).has_vector]
    return np.mean(vectors, axis=0) if vectors else None


# 获取单词的词向量
def get_word_vector(word):
    token = nlp(word)
    return token.vector if token.has_vector else None


# 计算余弦相似度的函数
def cosine_similarity(vec1, vec2):
    if vec1 is not None and vec2 is not None:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return -1  # 返回-1作为无法计算的情况


# 计算两组词之间的平均相似度
def get_average_similarity(group1, group2):
    similarities = []
    for word1 in group1:
        vec1 = get_word_vector(word1)
        for word2 in group2:
            vec2 = get_word_vector(word2)
            similarity = cosine_similarity(vec1, vec2)
            if similarity != -1:
                similarities.append(similarity)
    return np.mean(similarities) if similarities else None


def get_mean_similarity(group1, group2):
    similarities = []
    vec1 = get_mean_vector(group1)
    vec2 = get_mean_vector(group2)
    similarity = cosine_similarity(vec1, vec2)
    if similarity != -1:
        similarities.append(similarity)
    return np.mean(similarities) if similarities else None


def read_snippets(file_path):
    snippets_dict = {}  # 创建一个空字典来保存 snippets 的值
    params_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                item = json.loads(line)  # 尝试解析每一行作为一个独立的 JSON 对象
                snippets = item['snippets']  # 提取 snippets 字段
                params = item['params']
                for param in params:
                    snippets = param + "\n" + snippets
                var_value = item['var']
                snippets_dict[snippets] = var_value  # 将 snippets 添加到列表中
                params_dict[snippets] = params
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return snippets_dict


# 读取各个文件并保存结果
zero2one_dict = read_snippets('../../features/zero2one.jsonl')
one2zero_dict = read_snippets('../../features/one2zero.jsonl')
random_dict = read_snippets('../../features/random.jsonl')


def get_importance_score(args, ptm_model, code, label, tokenizer, type):
    temp_code_js = {"func": code, "target": label}
    feature = convert_examples_to_features(temp_code_js, tokenizer, args)
    dataset = CodeDataset([feature])
    logits, preds = ptm_model.get_results(dataset, args.eval_batch_size)
    orig_score = max(logits[0])
    orig_label = preds[0]

    lines = code.split("\n")
    importance_scores = []

    for i, line in enumerate(lines):
        new_code = lines[:]
        if type == 'replace':
            tokens = tokenizer.tokenize(line)
            masked_line = ' '.join(['<unk>' for _ in tokens])
            new_code[i] = masked_line
        elif type == 'insert':
            new_code[i] = line + '\n' + ' '.join(['<unk>'] * 10)

        new_code_str = "\n".join(new_code)
        new_code_js = {"func": new_code_str, "target": label}
        new_feature = convert_examples_to_features(new_code_js, tokenizer, args)
        new_dataset = CodeDataset([new_feature])
        new_logits, new_preds = ptm_model.get_results(new_dataset, args.eval_batch_size)
        new_score = max(new_logits[0])
        new_label = new_preds[0]
        score_difference = (orig_score - new_score).sum().item() if new_label == orig_label else 1

        importance_scores.append((i, score_difference))

    # Sort by importance score in descending order and extract line numbers
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_line_numbers = [x[0] for x in importance_scores]

    return sorted_line_numbers


def insert_codes(code, sorted_insert_code, insert_position):
    adv_codes = []
    insert_code_list = []
    for insert_code in sorted_insert_code:
        adv_code = insert_at_line(code, insert_position, insert_code)
        if adv_code not in adv_codes:
            adv_codes.append(adv_code)
            insert_code_list.append(insert_code)

    return adv_codes, insert_code_list

def transform_code(code, label, insert_position):
    new_codes = []
    for i in range(7):
        new_code=mutation(code, i, insert_position)
        if new_code != "":
            new_codes.append(new_code)
    new_codes = set(new_codes)
    new_codes = list(new_codes)
    return new_codes


class SLODA_Attacker():
    def __init__(self, args, ptm_model, model_tgt, tokenizer_tgt) -> None:
        self.args = args
        self.ptm_model = ptm_model
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt

    def code_trans_attack(self, code, true_label, orig_probs, orig_pred):
        query_times = 0
        is_success = -1
        adv_codes = []
        results = []
        orig_prob = orig_probs[true_label]
        insert_positions = get_importance_score(self.args, self.model_tgt, code, true_label,
                                                self.tokenizer_tgt, "replace")
        begin_line = 0
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if '{' in line.strip():
                begin_line = i + 1
                break
        valid_positions = list(range(begin_line, len(lines) - 1))
        # print(insert_positions)
        for i, insert_position in enumerate(insert_positions):
            if i >= 30:
                break
            if insert_position not in valid_positions:
                continue

            new_codes = transform_code(code, true_label, insert_position)
            # print(insert_position)
            try:
                new_codes.remove(code)
            except:
                pass
            for new_code in new_codes:
                # print(new_code)
                temp_code_js = {"func": new_code, "target": true_label}
                new_feature = convert_examples_to_features(temp_code_js, self.tokenizer_tgt, self.args)
                new_dataset = None
                if self.ptm_model == "GraphCodeBERT":
                    new_dataset = GraphCodeDataset([new_feature])
                elif self.ptm_model == "EPVD":
                    new_dataset = EPVDDataset([new_feature])
                else:
                    new_dataset = CodeDataset([new_feature])
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                temp_label = preds[0]
                temp_probs = logits[0]
                temp_prob = temp_probs[true_label]
                query_times += 1
                if temp_label != true_label:
                    is_success = 1
                    results.append(1)
                    print("%s SUC! (%.5f => %.5f)" % \
                          ('>>', true_label, temp_label), flush=True)
                    return is_success, adv_codes, new_code, query_times
                else:
                    results.append(0)
                # if temp_prob < orig_prob, add this code to adv_codes
                if temp_prob < orig_prob:
                    adv_codes.append(new_code)
                    if len(adv_codes) >= 100:
                        return is_success, adv_codes, None, query_times

        return is_success, adv_codes, None, query_times

    def taa_attack(self, true_label, id2token, code_lines, adv_codes, query):
        query_times = query
        is_success = -1
        existing_insert_code = []
        code_dict = zero2one_dict if true_label == 0 else one2zero_dict
        identifiers = get_identifiers(adv_codes[0])
        similarities = {}
        simi_time = time.time()
        random_keys = random.sample(list(code_dict.keys()), 100)
        random_dict = {key: code_dict[key] for key in random_keys}
        for key, value in random_dict.items():
            similarity = get_mean_similarity(value, identifiers)
            similarities[key] = similarity
        print(time.time() - simi_time)
        filtered_similarities = {k: v for k, v in similarities.items() if v is not None}
        # 按相似度排序并选择最高的30个
        sorted_insert_codes = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)[:30]
        for adv_code in adv_codes:

            sorted_line_numbers = get_importance_score(self.args, self.model_tgt, adv_code, true_label,
                                                       self.tokenizer_tgt, "insert")
            begin_line = 0
            lines = adv_code.split('\n')
            for i, line in enumerate(lines):
                if '{' in line.strip():
                    begin_line = i + 1
                    break
            valid_positions = list(range(begin_line, len(lines) - 1))

            best_code = None
            best_insert_code = None
            best_logits = None
            for i, position in enumerate(sorted_line_numbers):
                if i >= 30:
                    break
                if position not in valid_positions:
                    continue
                new_adv_codes, insert_code_list = insert_codes(adv_code, sorted_insert_codes, position)

                for tmp_code, insert_code in zip(new_adv_codes, insert_code_list):
                    temp_code_js = {"func": tmp_code, "target": true_label}
                    new_feature = convert_examples_to_features(temp_code_js, self.tokenizer_tgt, self.args)
                    new_dataset = None
                    if self.ptm_model == "GraphCodeBERT":
                        new_dataset = GraphCodeDataset([new_feature])
                    elif self.ptm_model == "EPVD":
                        new_dataset = EPVDDataset([new_feature])
                    else:
                        new_dataset = CodeDataset([new_feature])
                    logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                    temp_label = preds[0]
                    query_times += 1
                    if temp_label != true_label:
                        is_success = 1
                        print("%s SUC! (%.5f => %.5f)" % ('>>', true_label, temp_label), flush=True)
                        return is_success, tmp_code, query_times
                    if query_times > 1500:
                        return is_success, None, query_times
                    # Update the best code based on logits
                    if best_logits is None or max(logits[0]) < best_logits:
                        if insert_code not in existing_insert_code:
                            best_logits = max(logits[0])
                            best_code = tmp_code
                            best_insert_code = insert_code

                # Update adv_code with the best found code at the current position
                if best_code:
                    adv_code = best_code
                    existing_insert_code.append(best_insert_code)

        return is_success, None, query_times
def get_space(line):
    count1 = 0
    count2 = 0
    for char in line:
        if char == ' ':
            count1 += 1
        else:
            break
    for char in line:
        if char == '\t':
            count2 += 1
    return count1, count2

def compute_fitness(ptm_model, chromesome, model_tgt, tokenizer_tgt, orig_prob, orig_label, true_label, code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "c")

    new_feature = convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)
    if ptm_model == "GraphCodeBERT":
        new_dataset = GraphCodeDataset([new_feature])
    elif ptm_model == "EPVD":
        new_dataset = EPVDDataset([new_feature])
    else:
        new_dataset = CodeDataset([new_feature])
    new_logits, preds = model_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]

# def convert_code_to_features(code, tokenizer, label, args):
#     code=' '.join(code.split())
#     code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
#     source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
#     source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
#     padding_length = args.block_size - len(source_ids)
#     source_ids+=[tokenizer.pad_token_id]*padding_length
#     return InputFeatures(source_tokens,source_ids, 0, label)

def get_importance_score(args, example, true_label, ptm_model, code, words_list: list, sub_words: list, variable_names: list, tgt_model,
                         tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_feature = convert_code_to_features(new_code, tokenizer, true_label, args)
        new_example.append(new_feature)

    if ptm_model == "GraphCodeBERT":
        new_dataset = GraphCodeDataset(new_example)
    elif ptm_model == "EPVD":
        new_dataset = EPVDDataset(new_example)
    else:
        new_dataset = CodeDataset(new_example)

    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.

    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

def masked_line(code, line):
    start_line = line[0]
    end_line = line[1]
    code = remove_comments_and_docstrings(code, "c")
    code_list = code.split('\n')
    new_code_list = []
    mask_str = "<mask>"
    for index, line in enumerate(code_list):
        if start_line <= index <= end_line:
            mask_num = len(get_snippet_token(line, 'c'))
            new_code_list.append(mask_num * mask_str)
        else:
            new_code_list.append(code_list[index])
    masked_code = '\n'.join(new_code_list)

    return masked_code


def masked_expr_del(code, snp):
    code = remove_comments_and_docstrings(code, "c")
    code_list = code.split('\n')
    new_code_list = []
    for index, line in enumerate(code_list):
        if snp.find(line.lstrip()) < 0:
            new_code_list.append(code_list[index])
    masked_code = '\n'.join(new_code_list)
    return masked_code


def masked_line_del(code, line):
    start_line = line[0]
    end_line = line[1]
    code = remove_comments_and_docstrings(code, "c")
    code_list = code.split('\n')
    new_code_list = []

    for index, line in enumerate(code_list):
        if start_line <= index <= end_line:
            continue
        else:
            new_code_list.append(code_list[index])
    masked_code = '\n'.join(new_code_list)

    return masked_code


def masked_expr(code, snp):
    code = remove_comments_and_docstrings(code, "c")
    code_list = code.split('\n')
    new_code_list = []
    mask_str = "<mask>"
    for index, line in enumerate(code_list):
        if snp.find(line.lstrip()) >= 0:
            mask_num = len(get_snippet_token(line, 'c'))
            new_code_list.append(mask_num * mask_str)
        else:
            new_code_list.append(code_list[index])
    masked_code = '\n'.join(new_code_list)

    return masked_code


def deal_with_preds(preds1, preds2, snippets, true_label):
    pre_snippets = []
    pre_index = []
    snp_len = len(snippets)
    for index, pred in enumerate(preds1):
        if pred == true_label:
            pre_snippets.append(snippets[index])
            pre_index.append(index)
    for index2, pred2 in enumerate(preds2):
        idx2 = index2 % snp_len
        idx1 = index2 // snp_len
        if idx2 in pre_index:
            continue
        if pred2 == true_label and preds1[idx1] != true_label:
            pre_snippets.append(snippets[idx2])
            pre_index.append(idx2)
    pre_snippets = set(pre_snippets)
    pre_snippets = list(pre_snippets)
    return pre_snippets


def deal_with_preds_three(preds1, preds2, preds3, snippets, true_label):
    pre_snippets = []
    pre_index = []
    snp_len = len(snippets)
    for index, pred in enumerate(preds1):
        if pred == true_label:
            pre_snippets.append(snippets[index])
            pre_index.append(index)
    for index2, pred2 in enumerate(preds2):
        idx2 = index2 % snp_len
        idx1 = index2 // snp_len
        if idx2 in pre_index:
            continue
        if pred2 == true_label and preds1[idx1] != true_label:
            pre_snippets.append(snippets[idx2])
            pre_index.append(idx2)
    for index3, pred3 in enumerate(preds3):
        idex3 = index3 % snp_len
        idex2 = (index3 // snp_len) % snp_len
        idex1 = index3 // (snp_len * snp_len)
        if idex3 in pre_index:
            continue
        if pred3 == true_label and preds2[idex1 * snp_len + idex2] != true_label:
            pre_snippets.append(snippets[idex3])
            pre_index.append(idex3)
    pre_snippets = set(pre_snippets)
    pre_snippets = list(pre_snippets)
    return pre_snippets
    
class MHM_Attacker():
    def __init__(self, args, model_tgt, _ptm_model, _token2idx, _idx2token) -> None:
        self.ptm_model = _ptm_model
        self.classifier = model_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def mcmc_random(self, tokenizer, code=None, _label=None, _n_candi=30,
                    _max_iter=100, _prob_threshold=0.95):
        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)

        uid = get_identifier_posistions_from_code(words, identifiers)

        if len(uid) <= 0:  # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = random.sample(self.idx2token, _n_candi)

        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1 + _max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID_random(tokenizer, _tokens=code, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")

            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])  # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])

                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM-Origin"}
        replace_info = {}
        nb_changed_pos = 0

        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            nb_changed_pos += len(uid[old_uids[uid_][-1]])

        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid,
                "score_info": res["old_prob"][0] - res["new_prob"][0], "nb_changed_var": len(old_uids),
                "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM-Origin"}

    def __replaceUID_random(self, tokenizer, _tokens, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]
        selected_uid = random.sample(list(substitute_dict.keys()), 1)[0]  # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi):  # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if is_valid_identifier(c):  # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c)
                    # for i in _uid[selected_uid]: # 依次进行替换.
                    #     if i >= len(candi_tokens[-1]):
                    #         break
                    #     candi_tokens[-1][i] = c # 替换为新的candidate.

            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                temp_code_js = {"func": tmp_code, "target": _label}
                new_feature = convert_examples_to_features(temp_code_js, tokenizer, self.args)
                new_example.append(new_feature)

            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(new_example)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(new_example)
            else:
                new_dataset = CodeDataset(new_example)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):  # Find a valid example
                if pred[i] != _label:  # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1 - prob[candi_idx][_label] + 1e-10) / (1 - prob[0][_label] + 1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':  # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r':  # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a':  # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)


class ALERT_Attacker():
    def __init__(self, args, _ptm_model, model_tgt, tokenizer_tgt, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.ptm_model = _ptm_model
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def ga_attack(self, example, code, substituions, initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[3].item() if self.ptm_model == "GraphCodeBERT" else example[1].item()
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers_with_tokens(code, 'c')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..

        variable_names = list(substituions.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]

        if len(variable_substitue_dict) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []

                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]

                # 原来是随机选择的，现在要找到改变最大的.
                for a_substitue in variable_substitue_dict[tgt_word]:
                    # a_substitue = a_substitue.strip()

                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_code = get_example(code, tgt_word, a_substitue)
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                new_dataset = CodeDataset(replace_examples)
                # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = compute_fitness(self.ptm_model, temp_chromesome, self.model_tgt, self.tokenizer_tgt,
                                                       max(orig_prob), orig_label, true_label, code,
                                                       names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(self.args.eval_batch_size):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability:  # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else:  # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)

            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, "c")
                _tmp_feature = convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue

            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(feature_list)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(feature_list)
            else:
                new_dataset = CodeDataset(feature_list)

            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], code, "c")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    return code, prog_length, adv_code, true_label, orig_label, mutate_preds[
                        index], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index]
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)

            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None

    def greedy_attack(self, example, code, substituions):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[3].item() if self.ptm_model == "GraphCodeBERT" else example[1].item()
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers_with_tokens(code, 'c')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..

        variable_names = list(substituions.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.sep_token]
        # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example,
                                                                                               true_label,
                                                                                               self.ptm_model,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]

            all_substitues = substituions[tgt_word]

            # 得到了所有位置的substitue，并使用set来去重

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                # temp_replace = copy.deepcopy(final_words)
                # for one_pos in tgt_positions:
                #     temp_replace[one_pos] = substitute

                substitute_list.append(substitute)
                # 记录了替换的顺序

                # 需要将几个位置都替换成sustitue_
                temp_code = get_example(final_code, tgt_word, substitute)

                new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, true_label, self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(replace_examples)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(replace_examples)
            else:
                new_dataset = CodeDataset(replace_examples)
            # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate)
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

    def get_code_snippet_one(self, code, true_label, lang, example):
        block_pos, block_snp, expr_snp, dec_list, param_list = get_all_snippet(code, lang)

        snippets = []
        result = []
        block_len = len(block_snp)
        expr_len = len(expr_snp)
        snp_len = block_len + expr_len
        index = [1] * block_len + [0] * expr_len
        snippets = block_snp + expr_snp
        index_taken = []
        if len(snippets) == 0:
            return result
        replace_examples = []
        replace_examples2 = []
        replace_examples3 = []

        for idx1, snp1 in enumerate(snippets):
            if index[idx1] == 1:
                # temp_code = masked_line_del(code, snp1)
                temp_code1 = masked_line(code, block_pos[idx1])
            else:
                # temp_code = masked_expr_del(code, snp1)
                temp_code1 = masked_expr(code, snp1)
            new_feature = convert_code_to_features(temp_code1, self.tokenizer_tgt, example[1].item(), self.args)
            replace_examples.append(new_feature)
        new_dataset1 = CodeDataset(replace_examples)
        logits1, preds1 = self.model_tgt.get_results(new_dataset1, self.args.eval_batch_size)

        for idx1, snp1 in enumerate(snippets):

            if index[idx1] == 1:
                # temp_code = masked_line_del(code, snp1)
                temp_code1 = masked_line(code, block_pos[idx1])
            else:
                # temp_code = masked_expr_del(code, snp1)
                temp_code1 = masked_expr(code, snp1)

            for idx2, snp2 in enumerate(snippets):
                if index[idx2] == 1:
                    # temp_code = masked_line_del(code, snp1)
                    temp_code2 = masked_line(temp_code1, block_pos[idx2])
                else:
                    # temp_code = masked_expr_del(code, snp1)
                    temp_code2 = masked_expr(temp_code1, snp2)
                new_feature = convert_code_to_features(temp_code2, self.tokenizer_tgt, example[1].item(), self.args)
                replace_examples2.append(new_feature)
        new_dataset2 = CodeDataset(replace_examples2)
        logits2, preds2 = self.model_tgt.get_results(new_dataset2, self.args.eval_batch_size)
        for idx1, snp1 in enumerate(snippets):
            if index[idx1] == 1:
                # temp_code = masked_line_del(code, snp1)
                temp_code1 = masked_line(code, block_pos[idx1])
            else:
                # temp_code = masked_expr_del(code, snp1)
                temp_code1 = masked_expr(code, snp1)
            for idx2, snp2 in enumerate(snippets):
                if index[idx2] == 1:
                    # temp_code = masked_line_del(code, snp1)
                    temp_code2 = masked_line(temp_code1, block_pos[idx2])
                else:
                    # temp_code = masked_expr_del(code, snp1)
                    temp_code2 = masked_expr(temp_code1, snp2)
                for idx3, snp3 in enumerate(snippets):
                    if index[idx3] == 1:
                        # temp_code = masked_line_del(code, snp1)
                        temp_code3 = masked_line(temp_code2, block_pos[idx3])
                    else:
                        # temp_code = masked_expr_del(code, snp1)
                        temp_code3 = masked_expr(temp_code2, snp3)
                    new_feature = convert_code_to_features(temp_code3, self.tokenizer_tgt, example[1].item(), self.args)
                    replace_examples3.append(new_feature)
        new_dataset3 = CodeDataset(replace_examples3)
        logits3, preds3 = self.model_tgt.get_results(new_dataset3, self.args.eval_batch_size)
        pre_snippets = deal_with_preds_three(preds1, preds2, preds3, snippets, true_label)
        if len(pre_snippets) != 0:
            for snp in pre_snippets:
                new_code, var_spc, params = add_dec(dec_list, param_list, snp, code)
                result.append((new_code, var_spc, params))
        return result

class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, ptm_model, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.ptm_model = ptm_model
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def wir_attack(self, example, true_label, code, label):
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        # 这里经过了小写处理..
        variable_names = identifiers
        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.sep_token]
        # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example,
                                                                                               true_label,
                                                                                               self.ptm_model,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names[:20]:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]

            all_substitues = []
            num = 0
            while num < 5:
                tmp_var = random.choice(self.idx2token)
                if is_valid_identifier(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            # 得到了所有位置的substitue，并使用set来去重

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                # temp_replace = copy.deepcopy(final_words)
                # for one_pos in tgt_positions:
                #     temp_replace[one_pos] = substitute

                substitute_list.append(substitute)
                # 记录了替换的顺序

                # 需要将几个位置都替换成sustitue_
                temp_code = get_example(final_code, tgt_word, substitute)
                temp_code_js = {"func": temp_code, "target": true_label}

                new_feature = convert_examples_to_features(temp_code_js, self.tokenizer_tgt, self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            new_dataset = None
            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(replace_examples)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(replace_examples)
            else:
                new_dataset = CodeDataset(replace_examples)
            # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate)
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words


class DIP_Attacker():
    def __init__(self, args, _ptm_model, model_tgt, tokenizer_tgt, tokenizer_mlm) -> None:
        self.args = args
        self.ptm_model = _ptm_model
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm

    def dip_attack(self, example, code, snippet, line_seq, true_label):
        is_success = -1
        identifiers = get_identifiers(code)

        for index, line in enumerate(line_seq):

            for snp in snippet:
                replace_examples = []
                code_list = code.split("\n")
                var = random.choice(identifiers)
                intv = random.randint(0, 100)
                new_code_list = copy.deepcopy(code_list)
                s1, s2 = get_space(new_code_list[line])
                insert_line = s1 * " " + s2 * "\t" + "string " + var + "_" + str(intv) + " = " + "\"" + snp + "\""
                new_code_list.insert(line, insert_line)
                new_code = "\n".join(new_code_list)
                if self.ptm_model == "GraphCodeBERT":
                    new_feature = convert_code_to_features(new_code, self.tokenizer_tgt, example[3].item(), self.args)
                    replace_examples.append(new_feature)
                else:
                    new_feature = convert_code_to_features(new_code, self.tokenizer_tgt, example[1].item(), self.args)
                    replace_examples.append(new_feature)
                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                if self.ptm_model == "GraphCodeBERT":
                    new_dataset = GraphCodeDataset(replace_examples)
                elif self.ptm_model == "EPVD":
                    new_dataset = EPVDDataset(replace_examples)
                else:
                    new_dataset = CodeDataset(replace_examples)
                # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                if preds[0] != true_label:
                    is_success = 1
                    return is_success, new_code

        return is_success, code
        
        
        
def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "c")
    _, _, code_tokens = get_identifiers_coda(new_code, "c")
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    sub_words = [tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings


class CODA_Attacker:
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.codebert_mlm = codebert_mlm
        self.fasttext_model = fasttext_model
        self.substitutions = generated_substitutions

    def attack(self, example, code,query):
        NUMBER_1 = 256
        NUMBER_2 = 64
        #print([example])
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        print("orig_label",orig_label)
        print("orig_prob",orig_prob)
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif self.args.model_name == 'unixcoder':
            true_label = example[1].item()
        variable_names, function_names, code_tokens = get_identifiers_coda(code, "c")
        if (not orig_label == true_label) or len(variable_names)+len(function_names) == 0:
            return -2, None, None,0
        all_variable_name = []
        random_subs = []
        all_code = [code] * NUMBER_2
        all_code_csc = [code] * NUMBER_2
        while len(random_subs) < NUMBER_1 and np.max(orig_prob) >= 0:
            
            orig_prob[np.argmax(orig_prob)] = -1
            topn_label = np.argmax(orig_prob)
            print("topn_label",topn_label)
            for i in np.random.choice(self.substitutions[str(topn_label)], size=len(self.substitutions[str(topn_label)]),
                                      replace=False):
                if len(i['variable_name']) < len(variable_names) or len(i['function_name']) < len(function_names):
                    continue
                all_variable_name.extend(i['variable_name'])
                temp = copy.deepcopy(i)
                temp['label'] = str(topn_label)
                random_subs.append(temp)
                if len(random_subs) >= NUMBER_1:
                    break
        substituions = []
        ori_embeddings = get_embeddings(code, variable_names+function_names, self.tokenizer_mlm, self.codebert_mlm)
        ori_embeddings = torch.nn.functional.pad(ori_embeddings, [0, 0, 0, 512 - np.shape(ori_embeddings)[1]])
        embeddings_leng = np.shape(ori_embeddings)[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        print("get here to read npy")
        for sub in random_subs:
            
            embeddings_index = sub['embeddings_index']
            if self.args.model_name in ['codebert']:
                embeddings = np.load('../../dataset/coda_preprocess/codebert_all_subs_Devign/%s_%s.npy' % (sub['label'], embeddings_index))
            elif self.args.model_name in ['graphcodebert']:
                embeddings = np.load('../../dataset/coda_preprocess/codebert_all_subs_Devign/%s_%s.npy' % (sub['label'], embeddings_index))
            elif self.args.model_name in ['unixcoder']:
                embeddings = np.load('../../dataset/coda_preprocess/codebert_all_subs_Devign/%s_%s.npy' % (sub['label'], embeddings_index))
            embeddings = torch.from_numpy(embeddings).cuda()
            embeddings = torch.nn.functional.pad(embeddings, [0, 0, 0, 512 - np.shape(embeddings)[1]])
            substituions.append(([sub['variable_name'], sub['function_name'], sub['code']],
                                 np.sum(cos(ori_embeddings, embeddings).cpu().numpy()) / embeddings_leng))
        substituions = sorted(substituions, key=lambda x: x[1], reverse=True)
        substituions = [x[0] for x in substituions[:NUMBER_2]]

        temp_subs_variable_name = set()
        temp_subs_function_name = set()
        subs_code = []
        for subs in substituions:
            for i in subs[0]:
                temp_subs_variable_name.add(i)
            for i in subs[1]:
                temp_subs_function_name.add(i)
            subs_code.append(subs[2])
        min_prob = current_prob

        code_style = get_code_style(subs_code, 'c')
        replace_examples = []
        all_code_new = []
        for temp in all_code_csc:
            try:
                temp_code = change_code_style(temp, "c", all_variable_name, code_style)[-1]
            except:
                temp_code = temp
            all_code_new.append(temp_code)
            if self.args.model_name == 'codebert':
                new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(),
                                                                self.args)
            elif self.args.model_name == 'graphcodebert':
                new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[3].item(),
                                                                     self.args)
            elif self.args.model_name == 'unixcoder':
                new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(),
                                                              self.args)
            replace_examples.append(new_feature)
        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples)
        elif self.args.model_name == 'unixcoder':
            new_dataset = CodeDataset(replace_examples)
        #print(new_dataset)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        query+=len(replace_examples)
        for index, temp_prob in enumerate(logits):
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', current_prob, temp_prob[orig_label]),
                      flush=True)
                return 2, all_code_new[index], current_prob - min(min_prob, temp_prob[orig_label]),query
            else:
                if min_prob > temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    code = all_code_new[index]
        print("%s FAIL! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)

        subs_variable_name = []
        subs_function_name = []
        for i in temp_subs_variable_name:
            subs_variable_name.append([i, self.fasttext_model.get_word_vector(i)])
        for i in temp_subs_function_name:
            subs_function_name.append([i, self.fasttext_model.get_word_vector(i)])
        substituions = {}
        for i in variable_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_variable_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        for i in function_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_function_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]

        all_code = []
        all_code_csc = []
        final_code = None
        replace_examples = []
        current_subs = ['' for i in range(len(variable_names) + len(function_names))]
        for i in range(NUMBER_2):
            temp_code = copy.deepcopy(code)
            for j, tgt_word in enumerate(variable_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j] = substituions[tgt_word][i]
                temp_code = get_example_coda(temp_code, tgt_word, substituions[tgt_word][i], "c")
                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                         example[3].item(), self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(),
                                                                  self.args)
                replace_examples.append(new_feature)

            for j, tgt_word in enumerate(function_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j + len(variable_names)] = substituions[tgt_word][i]
                temp_code = get_example_coda(temp_code, tgt_word, substituions[tgt_word][i], "c")
                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                         example[3].item(), self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                  example[1].item(), self.args)
                replace_examples.append(new_feature)
            try:
                all_code_csc.append(all_code[-1])
            except:
                pass
        if len(replace_examples) == 0:
            return -3, None, None,None
        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples)
        elif self.args.model_name == 'unixcoder':
            new_dataset = CodeDataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        query+=len(replace_examples)
        for index, temp_prob in enumerate(logits):
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', current_prob, temp_prob[orig_label]),
                      flush=True)
                return 1, all_code[index], current_prob - temp_prob[orig_label],query
            else:
                if min_prob >= temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    final_code = all_code[index]
        print("%s FAIL! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)

        return -1, final_code, current_prob - min_prob,query
        


