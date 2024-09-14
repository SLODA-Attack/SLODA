import sys
import os
import time

sys.path.append('../../dataset/')
sys.path.append('../CodeBERT/code/')   # CodeBERT, UniXcoder, GraphCodeBERT, EPVD, ReGVD
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

class ALERT_Attacker():
    def __init__(self, args, _ptm_model, model_tgt, tokenizer_tgt, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.ptm_model = _ptm_model
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

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