import sys
import os

sys.path.append('../dataset/')
sys.path.append('../dataset/python_parser')
from data_augmentation import get_identifiers, get_example,get_code_tokens
import copy
import time
import spacy
from run_parser import get_identifiers_with_tokens,mutation,get_snippet_token
from data_utils import is_valid_identifier
from attack_utils import get_masked_code_by_position, _tokenize, get_llm_result,insert_at_line
from attack_utils import map_chromesome, CodeDataset, GraphCodeDataset, get_identifier_posistions_from_code
import random
import re
import json
from utils import select_parents, crossover, map_chromesome, mutate
from utils import select_parents, crossover, map_chromesome, mutate
from python_parser.run_parser_CODA import get_identifiers_coda,get_example_batch,get_code_style, change_code_style,get_example_coda
from scipy.spatial.distance import cosine as cosine_distance
import torch

import numpy as np
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

def compute_fitness(chromesome, model_tgt, tokenizer_tgt, orig_prob, orig_label, code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "c")

    prob, pred, _ = get_llm_result(temp_code, model_tgt, tokenizer_tgt, orig_label)

    # 计算fitness function
    fitness_value = orig_prob - prob
    return fitness_value, pred

def get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model,
                         tokenizer, orig_label, label_list, batch_size=16, max_length=512, model_type='classification'):
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
    probs = []
    preds = []
    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        prob, result, flag = get_llm_result(new_code, tgt_model, tokenizer, orig_label)
        probs.append(prob)
        preds.append(result)

    orig_prob = probs[0]
    orig_label = preds[0]
    # predicted label对应的probability

    importance_score = []
    for prob in probs[1:]:
        importance_score.append(orig_prob - prob)

    return importance_score, replace_token_positions, positions, len([words_list] + masked_token_list)

class MHM_Attacker():
    def __init__(self, args, model_tgt, tokenizer, _token2idx, _idx2token) -> None:
        self.model = model_tgt
        self.tokenizer = tokenizer
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def mcmc_random(self, code=None, _label=None, _n_candi=30,
                    _max_iter=100, _prob_threshold=0.95):
        print("label: ", _label)
        query = 0
        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer)
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
            res = self.__replaceUID_random(_tokens=code, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            query += _n_candi
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
                    return {'succ': True, "query": query, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid,
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM"}
        replace_info = {}
        nb_changed_pos = 0

        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            nb_changed_pos += len(uid[old_uids[uid_][-1]])

        return {'succ': False, "query": query, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid,
                "nb_changed_var": len(old_uids),
                "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM"}

    def __replaceUID_random(self, _tokens, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]

        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名
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
            probs = []
            preds = []
            for tmp_code in candi_tokens:
                prob, result, _ = get_llm_result(tmp_code, self.model, self.tokenizer, _label)
                preds.append(result)
                probs.append(prob)
            for i in range(len(candi_token)):  # Find a valid example
                if preds[i] != _label and preds[i] in ['0', '1']:  # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": probs[0], "new_prob": probs[i],
                            "old_pred": preds[0], "new_pred": preds[i], "nb_changed_pos": _tokens.count(selected_uid)}

            # candi_idx = 0
            # min_prob = 1.0
            #
            # for idx, a_prob in enumerate(probs):
            #     if a_prob < min_prob:
            #         candi_idx = idx
            #         min_prob = a_prob

            candi_idx = probs.index(min(probs))
            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1 - probs[candi_idx] + 1e-10) / (1 - probs[0] + 1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": probs[0], "new_prob": probs[i],
                        "old_pred": preds[0], "new_pred": preds[i], "nb_changed_pos": _tokens.count(selected_uid)}
            elif preds[i] in ['0', '1']:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": probs[0], "new_prob": probs[i],
                        "old_pred": preds[0], "new_pred": preds[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':  # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   int(_res['old_pred']), int(_res['new_pred']),
                   _res['old_prob'],
                   _res['new_prob'], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r':  # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   int(_res['old_pred']), int(_res['new_pred']),
                   _res['old_prob'],
                   _res['new_prob'], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a':  # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   int(_res['old_pred']), int(_res['new_pred']),
                   _res['old_prob'],
                   _res['new_prob'], _res['alpha']), flush=True)


class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def wir_attack(self, example, code, label, prob):
        query = 0
        orig_label = label
        current_prob = prob

        true_label = label
        adv_code = ''
        temp_label = None

        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        # 这里经过了小写处理..

        variable_names = identifiers
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, query, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.bos_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.eos_token]
        # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict, score_query = get_importance_score(self.args, example,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               true_label,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')
        query += score_query
        if importance_score is None:
            return code, prog_length, adv_code, query, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

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

        for name_and_score in sorted_list_of_names[:17]:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]

            all_substitues = []
            num = 0
            while num < 30:
                tmp_var = random.choice(self.idx2token)
                if is_valid_identifier(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            # 得到了所有位置的substitue，并使用set来去重

            most_gap = 0.0
            candidate = None
            preds = []
            probs = []
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
                prob, result, _ = get_llm_result(temp_code, self.model_tgt, self.tokenizer_tgt, true_label)
                query += 1
                preds.append(result)
                probs.append(prob)

            for index, temp_prob in enumerate(probs):
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
                           temp_prob), flush=True)
                    return code, prog_length, adv_code, query, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob
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

        return code, prog_length, adv_code, query, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

class ALERT_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def ga_attack(self, code, query, substituions, label, initial_replace=None):
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

        current_prob, orig_label, _ = get_llm_result(code, self.model_tgt, self.tokenizer_tgt, label)

        orig_prob = current_prob
        query += 1
        true_label = orig_label
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers_with_tokens(code, 'c')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        # 这里经过了小写处理..

        variable_names = list(substituions.keys())

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
            return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, None, None, None, None

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None or tgt_word not in initial_replace:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []

                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]

                # 原来是随机选择的，现在要找到改变最大的.
                _the_best_candidate = -1
                logits = []
                preds = []
                for a_substitue in variable_substitue_dict[tgt_word]:
                    # a_substitue = a_substitue.strip()

                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_code = get_example(code, tgt_word, a_substitue)
                    prob, pred, _ = get_llm_result(temp_code, self.model_tgt, self.tokenizer_tgt, label)
                    query += 1
                    if query >= 500:
                        is_success = -1
                        return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, None, None, None, None
                    logits.append(prob)
                    preds.append(pred)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    gap = current_prob - temp_prob

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
            temp_fitness, temp_label = compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt,
                                                       orig_prob, orig_label, code,
                                                       names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.
        for i in range(max_iter):
            _temp_mutants = []
            for j in range(64):
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

            if len(_temp_mutants) == 0:
                continue
            # compute fitness in batch
            feature_list = []
            mutate_logits = []
            mutate_preds = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, "c")
                prob, pred, _ = get_llm_result(_temp_code, self.model_tgt, self.tokenizer_tgt, label)
                query += 1
                if query >= 500:
                    is_success = -1
                    return code, prog_length, adv_code, query, orig_label, temp_label, is_success, variable_names, None, None, None, None

                mutate_logits.append(prob)
                mutate_preds.append(mutate_preds)

            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], code, "c")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    return code, prog_length, adv_code, query, true_label, mutate_preds[
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

        return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None

    def greedy_attack(self, example, code, label, prob, substituions):
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
        query = 0
        orig_label = label
        current_prob = prob

        true_label = label
        adv_code = ''
        temp_label = None

        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        # 这里经过了小写处理..

        variable_names = list(substituions.keys())

        sub_words = [self.tokenizer_tgt.bos_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.eos_token]
        # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict, query_score = get_importance_score(self.args, example,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               true_label,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')
        query += query_score
        if importance_score is None:
            return code, prog_length, adv_code, query, true_label, temp_label, -3, variable_names, None, None, None, None

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
            preds = []
            probs = []
            for substitute in all_substitues[:30]:
                # temp_replace = copy.deepcopy(final_words)
                # for one_pos in tgt_positions:
                #     temp_replace[one_pos] = substitute

                substitute_list.append(substitute)
                # 记录了替换的顺序

                # 需要将几个位置都替换成sustitue_
                temp_code = get_example(final_code, tgt_word, substitute)
                prob, result, _ = get_llm_result(temp_code, self.model_tgt, self.tokenizer_tgt, label)
                query += 1
                if query >= 500:
                    # 说明原来就是错的
                    is_success = -1
                    return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, None, None, None, None

                preds.append(result)
                probs.append(prob)

            for index, temp_prob in enumerate(probs):
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
                           temp_prob), flush=True)
                    return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob
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

        return code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

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
zero2one_dict = read_snippets('../features/zero2one.jsonl')
one2zero_dict = read_snippets('../features/one2zero.jsonl')
random_dict = read_snippets('../features/random.jsonl')


def get_importance_score_taa(args, ptm_model, code, label, tokenizer, type):
    prob, result, _ = get_llm_result(code, ptm_model, tokenizer, label)

    orig_score = prob
    orig_label = result

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
        prob, result, _ = get_llm_result(new_code_str, ptm_model, tokenizer, label)
        new_score = prob
        new_label = result
        score_difference = (orig_score - new_score) if new_label == orig_label else 1

        importance_scores.append((i, score_difference))

    # Sort by importance score in descending order and extract line numbers
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_line_numbers = [x[0] for x in importance_scores]

    return sorted_line_numbers
    
def get_mean_similarity(group1, group2):
    similarities = []
    vec1 = get_mean_vector(group1)
    vec2 = get_mean_vector(group2)
    similarity = cosine_similarity(vec1, vec2)
    if similarity != -1:
        similarities.append(similarity)
    return np.mean(similarities) if similarities else None

def insert_codes(code, sorted_insert_code, insert_position):
    adv_codes = []
    insert_code_list = []
    #print("code",code)
    #print("sorted_insert_code num: ",len(sorted_insert_code))
    
    #for index ,insert_code in enumerate(sorted_insert_code):
        #print("index: ",index,"insert_code: ",insert_code)
        
    for index ,insert_code in enumerate(sorted_insert_code):
        adv_code = insert_at_line(code, insert_position, insert_code)
        #print("adv_code: ",adv_code)
        if adv_code not in adv_codes:
            adv_codes.append(adv_code)
            #print(len(adv_codes))
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


class TAA_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt

    def code_trans_attack(self, code, true_label, orig_prob, orig_pred):
        query_times = 0
        is_success = -1
        adv_codes = []
        results = []
        begin_line = 0
        lines = code.split('\n')
        pos_invalid=[]
        for i, line in enumerate(lines):
            if '{' in line.strip():
                begin_line = i + 1
                break
        for i, line in enumerate(lines):
            if lines[i].strip() == '':
                pos_invalid.append(i)

        valid_positions = list(range(begin_line, len(lines) - 2))

        for i in pos_invalid:
            if i in valid_positions:
                valid_positions.remove(i)

        # print("valid_positions",valid_positions)
        insert_positions = get_importance_score_taa(self.args, self.model_tgt, code, true_label,
                                                    self.tokenizer_tgt, "replace")
        # print(insert_positions)
        for i,insert_position in enumerate(insert_positions):
            if i>=15:
                break;
            if insert_position not in valid_positions:
                continue
            new_codes = transform_code(code, true_label, insert_position)
            # print(insert_position)
            try:
                new_codes.remove(code)
            except:
                pass
            for new_code in new_codes:
                prob, result, _ = get_llm_result(new_code, self.model_tgt, self.tokenizer_tgt, true_label)
                query_times += 1
                if query_times >= 500:
                    is_success = -1
                    return is_success,None,None, query_times
                temp_label = result
                temp_prob = prob
                if temp_label != true_label:
                    is_success = 1
                    results.append(1)
                    
                    return is_success, adv_codes, new_code, query_times
                else:
                    results.append(0)
                # if temp_prob < orig_prob, add this code to adv_codes
                if temp_prob < orig_prob:
                    adv_codes.append(new_code)
                    if len(adv_codes) >= 20:
                        return is_success, adv_codes, None, query_times

        return is_success, adv_codes, None, query_times

    def taa_attack(self, true_label, adv_codes,query):
        query_times = query
        is_success = -1
        existing_insert_code = []
        code_dict = zero2one_dict if true_label == 0 else one2zero_dict
        identifiers = get_identifiers(adv_codes[0])
        similarities = {}
        simi_time=time.time()
        random_keys = random.sample(list(code_dict.keys()), 100)
        for key in random_keys:
            snippet_token = get_snippet_token(key,"c")
            if len(snippet_token)>100:
                random_keys.remove(key)
        random_dict = {key: code_dict[key] for key in random_keys}
        for key, value in random_dict.items():
            similarity = get_mean_similarity(value, identifiers)
            similarities[key] = similarity
        #print(time.time()-simi_time)
        filtered_similarities = {k: v for k, v in similarities.items() if v is not None}
        # 按相似度排序并选择最高的30个
        sorted_insert_codes = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)[:30]
        #print("sorted_insert_codes: ",sorted_insert_codes)
        for adv_code in adv_codes:
            sorted_line_numbers = get_importance_score_taa(self.args, self.model_tgt, adv_code, true_label,
                                                           self.tokenizer_tgt, "insert")
            begin_line = 0
            lines = adv_code.split('\n')
            pos_invalid = []
            for i, line in enumerate(lines):
                if '{' in line.strip():
                    begin_line = i + 1
                    break
            for i, line in enumerate(lines):
                if lines[i].strip() == '':
                    pos_invalid.append(i)

            valid_positions = list(range(begin_line, len(lines) - 2))

            for i in pos_invalid:
                if i in valid_positions:
                    valid_positions.remove(i)

            best_code = None
            best_insert_code = None
            best_logits = None
            # print("valid_positions: ",valid_positions)
            # print("sorted_line_numbers: ",sorted_line_numbers)
            for i,position in enumerate(sorted_line_numbers):
                # print("line_number: ",i,"position",position)
                if i>=5:
                    break
                if position not in valid_positions:
                    continue
                new_adv_codes, insert_code_list = insert_codes(adv_code, sorted_insert_codes, position)

                for tmp_code, insert_code in zip(new_adv_codes, insert_code_list):
                    prob, result, _ = get_llm_result(tmp_code, self.model_tgt, self.tokenizer_tgt, true_label)
                    query_times += 1
                    if query_times >= 500:
                        is_success = -1
                        return is_success, None, query_times
                    temp_prob = prob
                    temp_label = result
                    if temp_label != true_label:
                        is_success = 1
                        
                        return is_success, tmp_code, query_times

                    # Update the best code based on logits
                    if best_logits is None or temp_prob < best_logits:
                        if insert_code not in existing_insert_code:
                            best_logits = temp_prob
                            best_code = tmp_code
                            best_insert_code = insert_code

                # Update adv_code with the best found code at the current position
                if best_code:
                    adv_code = best_code
                    existing_insert_code.append(best_insert_code)

        return is_success, None, query_times



class TAA_Attacker_LOF():
    def __init__(self, args, model_tgt, tokenizer_tgt) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt

    def code_trans_attack(self, code, true_label, orig_prob, orig_pred):
        query_times = 0
        is_success = -1
        adv_codes = []
        results = []
        begin_line = 0
        lines = code.split('\n')
        pos_invalid=[]
        for i, line in enumerate(lines):
            if '{' in line.strip():
                begin_line = i + 1
                break
        for i, line in enumerate(lines):
            if lines[i].strip() == '':
                pos_invalid.append(i)

        valid_positions = list(range(begin_line, len(lines) - 2))

        for i in pos_invalid:
            if i in valid_positions:
                valid_positions.remove(i)

        #print("valid_positions",valid_positions)
        insert_positions = get_importance_score_taa(self.args, self.model_tgt, code, true_label,
                                                    self.tokenizer_tgt, "replace")
        # print(insert_positions)
        for i,insert_position in enumerate(insert_positions):
            if i>=15:
                break;
            if insert_position not in valid_positions:
                    continue
            new_codes = transform_code(code, true_label, insert_position)
            # print(insert_position)
            try:
                new_codes.remove(code)
            except:
                pass
            for new_code in new_codes:
                prob, result, _ = get_llm_result(new_code, self.model_tgt, self.tokenizer_tgt, true_label)
                query_times += 1
                if query_times >= 500:
                    is_success = -1
                    return is_success,None,None, query_times
                temp_label = result
                temp_prob = prob
                if temp_label != true_label:
                    is_success = 1
                    results.append(1)
                    
                    return is_success, adv_codes, new_code, query_times
                else:
                    results.append(0)
                # if temp_prob < orig_prob, add this code to adv_codes
                if temp_prob < orig_prob:
                    adv_codes.append(new_code)
                    if len(adv_codes) >= 20:
                        return is_success, adv_codes, None, query_times

        return is_success, adv_codes, None, query_times

    def taa_attack(self, true_label, adv_codes,query):
        query_times = query
        is_success = -1
        existing_insert_code = []
        code_dict = zero2one_dict if true_label == 0 else one2zero_dict
        identifiers = get_identifiers(adv_codes[0])
        similarities = {}
        simi_time=time.time()
        random_keys = random.sample(list(code_dict.keys()), 100)
        for key in random_keys:
            snippet_token = get_snippet_token(key,"c")
            if len(snippet_token)>100:
                random_keys.remove(key)
        random_dict = {key: code_dict[key] for key in random_keys}
        for key, value in random_dict.items():
            similarity = get_mean_similarity(value, identifiers)
            similarities[key] = similarity
        #print(time.time()-simi_time)
        filtered_similarities = {k: v for k, v in similarities.items() if v is not None}
        # 按相似度排序并选择最高的30个
        sorted_insert_codes = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)[:20]
        
        for adv_code in adv_codes:
            #sorted_line_numbers = get_importance_score_taa(self.args, self.model_tgt, adv_code, true_label,
                                                           #self.tokenizer_tgt, "insert")
            #sorted_line_numbers = random.sample(valid_positions, len(valid_positions))
            begin_line = 0
            lines = adv_code.split('\n')
            pos_invalid = []
            for i, line in enumerate(lines):
                if '{' in line.strip():
                    begin_line = i + 1
                    break
            for i, line in enumerate(lines):
                if lines[i].strip() == '':
                    pos_invalid.append(i)

            valid_positions = list(range(begin_line, len(lines) - 2))
            sorted_line_numbers = random.sample(valid_positions, len(valid_positions))
            for i in pos_invalid:
                if i in valid_positions:
                    valid_positions.remove(i)

            best_code = None
            best_insert_code = None
            best_logits = None
            #print("valid_positions: ",valid_positions)
            #print("sorted_line_numbers: ",sorted_line_numbers)
            readed_line = 0
            for i,position in enumerate(sorted_line_numbers):
                #print("line_number: ",i,"position",position)
                if readed_line>=5:
                    break
                if position not in valid_positions:
                    continue
                
                new_adv_codes, insert_code_list = insert_codes(adv_code, sorted_insert_codes, position)
                #print(len(new_adv_codes))
                if len(new_adv_codes)!=1:
                    readed_line +=1

                for tmp_code, insert_code in zip(new_adv_codes, insert_code_list):
                    prob, result, _ = get_llm_result(tmp_code, self.model_tgt, self.tokenizer_tgt, true_label)
                    query_times += 1
                    if query_times >= 500:
                        is_success = -1
                        return is_success, None, query_times
                    temp_prob = prob
                    temp_label = result
                    if temp_label != true_label:
                        is_success = 1
                        
                        return is_success, tmp_code, query_times

                    # Update the best code based on logits
                    if best_logits is None or temp_prob < best_logits:
                        if insert_code not in existing_insert_code:
                            best_logits = temp_prob
                            best_code = tmp_code
                            best_insert_code = insert_code

                # Update adv_code with the best found code at the current position
                if best_code:
                    adv_code = best_code
                    existing_insert_code.append(best_insert_code)

        return is_success, None, query_times



class DIP_Attacker():
    def __init__(self, args, query,model_tgt, tokenizer_tgt) -> None:
        self.args = args

        self.query = query
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt


    def dip_attack(self, code, snippet, line_seq, true_label):
        is_success = -1
        identifiers = get_identifiers(code)
        query_times = 0
        #print(f"total query times : {len(line_seq)*len(snippet)} len(line_seq) : {len(line_seq)} len(snippet) : {len(snippet)}")
        for index, line in enumerate(line_seq):
            if index >=10:
                continue
            #print("line_num: ",line)
            for snp in snippet:
                #print("snp: ",line)
                code_list = code.split("\n")
                var = random.choice(identifiers)
                intv = random.randint(0, 100)
                new_code_list = copy.deepcopy(code_list)
                s1, s2 = get_space(new_code_list[line])
                insert_line = s1 * " " + s2 * "\t" + "string " + var + "_" + str(intv) + " = " + "\"" + snp + "\""
                new_code_list.insert(line, insert_line)
                new_code = "\n".join(new_code_list)
                prob, result, _ = get_llm_result(new_code, self.model_tgt, self.tokenizer_tgt, true_label)
                query_times += 1
                if query_times >= 500:
                    # 说明原来就是错的
                    is_success = -1
                    return is_success, None, query_times
                if result != true_label:
                    is_success = 1
                    return is_success, new_code, query_times

        return is_success, code, query_times
        
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
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, codebert_mlm, fasttext_model,
                 generated_substitutions) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.codebert_mlm = codebert_mlm
        self.fasttext_model = fasttext_model
        self.substitutions = generated_substitutions

    def attack(self, example, code, query):
        NUMBER_1 = 256
        NUMBER_2 = 64
        # print([example])
        prob, result, flag = get_llm_result(code, self.model_tgt, self.tokenizer_tgt, 0)
        orig_prob = prob
        orig_label = result
        #print("orig_label", orig_label)
        #print("orig_prob", orig_prob)

        true_label = example["output"]
        #print("true_label",true_label)
        #print("flag",flag)
        #print()
        query_times =0
        variable_names, function_names, code_tokens = get_identifiers_coda(code, "c")
        if (not orig_label == true_label) or len(variable_names) + len(function_names)  == 0 or flag==-1:
            
            return -2, None, None, 0
        all_variable_name = []
        random_subs = []
        all_code = [code] * NUMBER_2
        all_code_csc = [code] * NUMBER_2
        while_time = 0
        while len(random_subs) < NUMBER_1 and while_time < 2:
            temp_label = orig_label
            #print("temp_label", temp_label)
            #print(type(temp_label))
            topn_label = -1
            if temp_label == '1':
                #print("first")
                topn_label =1
                while_time+= 1
                temp_label = '0'
            elif temp_label == '0':
                #print("secend")
                topn_label = 0
                while_time += 1
                temp_label = '1'
            #print("topn_label", topn_label)
            for i in np.random.choice(self.substitutions[str(topn_label)],
                                      size=len(self.substitutions[str(topn_label)]),
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
        ori_embeddings = get_embeddings(code, variable_names + function_names, self.tokenizer_mlm, self.codebert_mlm)
        ori_embeddings = torch.nn.functional.pad(ori_embeddings, [0, 0, 0, 512 - np.shape(ori_embeddings)[1]])
        embeddings_leng = np.shape(ori_embeddings)[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        #print("get here to read npy")
        for sub in random_subs:

            embeddings_index = sub['embeddings_index']
            embeddings = np.load('./codebert_all_subs_Devign/%s_%s.npy' % (sub['label'], embeddings_index))
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
        min_prob = orig_prob

        code_style = get_code_style(subs_code, 'c')
        replace_examples = []
        all_code_new = []
        for temp in all_code_csc:
            try:
                temp_code = change_code_style(temp, "c", all_variable_name, code_style)[-1]
            except:
                temp_code = temp
            all_code_new.append(temp_code)

        for temp_code in all_code_new:
            prob, result, _ = get_llm_result(temp_code, self.model_tgt, self.tokenizer_tgt, true_label)
            query_times += 1
            if query_times >= 300:
                is_success = -1
                return is_success, None,None, query_times
            temp_prob = prob
            temp_label = result
            if temp_label != true_label:
                is_success = 2
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', orig_prob, temp_prob),
                      flush=True)
                return is_success, temp_code, None,query_times
            else:
                if min_prob > temp_prob:
                    min_prob = temp_prob
                    code = temp_code
        print("%s FAIL! (%.5f => %.5f)" % ('>>', orig_prob, min_prob), flush=True)


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


            for j, tgt_word in enumerate(function_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j + len(variable_names)] = substituions[tgt_word][i]
                temp_code = get_example_coda(temp_code, tgt_word, substituions[tgt_word][i], "c")
                all_code.append(temp_code)
            try:
                all_code_csc.append(all_code[-1])
            except:
                pass

        for temp_code in all_code:
            prob, result, _ = get_llm_result(temp_code, self.model_tgt, self.tokenizer_tgt, true_label)
            query_times += 1
            if query_times >= 300:
                is_success = -1
                return is_success, None,None, query_times
            temp_prob = prob
            temp_label = result
            if temp_label != true_label:
                is_success = 1
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', orig_prob, temp_prob),
                      flush=True)
                return is_success, temp_code, None,query_times
            else:
                if min_prob > temp_prob:
                    min_prob = temp_prob

        print("%s FAIL! (%.5f => %.5f)" % ('>>', orig_prob, min_prob), flush=True)


        return -1, final_code, orig_prob - min_prob, query