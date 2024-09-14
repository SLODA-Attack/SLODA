import sys
import os

sys.path.append('../../dataset/')
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import warnings
import time
from attack_utils import build_vocab, set_seed
from attack_utils import Recorder, get_llm_result, Recorder_style
from llm_attacker import DIP_Attacker
import torch
import os
import sys
import json
import copy
import random
import pandas as pd
import argparse
from peft import PeftModel
import numpy as np
from transformers import GenerationConfig
from transformers import LlamaTokenizer, CodeLlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaForSequenceClassification,
                          RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    if vec1 is not None and vec2 is not None:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return -1  # 返回-1作为无法计算的情况


def get_dip_sub(code):
    code_list = code.split('\n')
    ans_code_list = []
    unk_line = "<unk>" * 10
    for index, line in enumerate(code_list):
        if index == 0 or index == len(code_list) - 1 or index == 1 or index == 2:
            continue
        new_code_list = copy.deepcopy(code_list)
        new_code_list.insert(index, unk_line)
        new_code = "\n".join(new_code_list)
        ans_code_list.append(new_code)
    return ans_code_list


def get_dip_snippet(code, tokenizer, model):
    lines = code.split('\n')

    # 对输入代码进行分词
    inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512)

    # 将tokens传入模型以获取注意力得分
    inputs = inputs.to('cuda')
    outputs = model(**inputs)


    # 提取注意力得分
    # outputs.attentions是一个包含每层注意力得分的元组
    # 倒数第二层的索引是-2
    attention_scores = outputs.attentions[-2]

    # 注意力得分的形状为(batch_size, num_heads, seq_length, seq_length)
    # 我们需要在heads维度上取平均值以获取每个token的注意力得分
    attention_scores = attention_scores.mean(dim=1).squeeze(0)  # 形状: (seq_length, seq_length)

    # 每个token的注意力得分可以通过在序列长度维度上取平均值或求和来获得
    token_attention_scores = attention_scores.mean(dim=0)  # 形状: (seq_length,)

    # 解码tokens以了解每个得分对应哪个token
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

    # 按行计算平均注意力得分
    line_attention_scores = []
    current_line_start = 1  # 跳过第一个[CLS] token
    for line in lines:
        line_tokens = tokenizer.tokenize(line)
        line_length = len(line_tokens)
        line_scores = token_attention_scores[current_line_start:current_line_start + line_length]
        line_average_score = line_scores.mean().item()
        line_attention_scores.append(line_average_score)
        current_line_start += line_length + 1  # 跳过[SEP] token

    max_score = max(line_attention_scores)
    max_score_index = line_attention_scores.index(max_score)
    max_score_line = lines[max_score_index]

    return max_score_line


def dip_comp_code(code1, code_list, line_flag, tokenizer, model):
    similarity = {}
    sim_line_dist = {}
    sorted_sim_dict = {}
    sorted_line_dict = {}
    if line_flag == True:
        for code2 in code_list:
            code2_list = code2.split("\n")
            # 得到<unk>行的行号
            unk_line_list = [index for index, line in enumerate(code2_list) if "<unk>" in line]
            if len(unk_line_list) != 1:
                continue
            unk_line = unk_line_list[0]
            inputs_original = tokenizer(code1, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_modified = tokenizer(code2, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_original = inputs_original.to('cuda')
            inputs_modified = inputs_modified.to('cuda')
            with torch.no_grad():
                outputs_original = model(**inputs_original)
                outputs_modified = model(**inputs_modified)
            cls_representation_original = outputs_original.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
            cls_representation_modified = outputs_modified.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
            Vp = cosine_similarity(cls_representation_original, cls_representation_modified)
            sim_line_dist[unk_line] = Vp
        sorted_line_dict = {k: v for k, v in sorted(sim_line_dist.items(), key=lambda item: item[1])}
        return sorted_line_dict
    else:
        for code2 in code_list:
            code2_list = code2.split("\n")
            # 得到<unk>行的行号
            inputs_original = tokenizer(code1, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_modified = tokenizer(code2, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_original = inputs_original.to('cuda')
            inputs_modified = inputs_modified.to('cuda')
            with torch.no_grad():
                outputs_original = model(**inputs_original)
                outputs_modified = model(**inputs_modified)
            cls_representation_original = outputs_original.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
            cls_representation_modified = outputs_modified.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
            Vp = cosine_similarity(cls_representation_original, cls_representation_modified)
            similarity[code2] = Vp
        sorted_sim_dict = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1])}
        return sorted_sim_dict


def print_index(index, label, orig_pred):
    entry = f"Index: {index} | label: {label} | orig_pred: {orig_pred}"
    entry_length = len(entry)
    padding = 100 - entry_length
    left_padding = padding // 2
    right_padding = padding - left_padding
    result = f"{'*' * left_padding} {entry} {'*' * right_padding}"
    print(result, flush=True)


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument('--head', type=int,
                        help="number of dataset examples to cut off")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument('--code_key', type=str, default="input",
                        help="dataset key for code")
    parser.add_argument('--label_key', type=str, default="output",
                        help="dataset key for labels")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--tuned_model", required=True, help="Path to the tuned model.")
    parser.add_argument("--eval_data_file", required=True, help="Path to the json file containing test dataset.")
    parser.add_argument("--csv_path", default='results.csv', help="Path to save the CSV results.")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    print("Base model loaded!")

    model = PeftModel.from_pretrained(model, args.tuned_model)
    print("Tuned model loaded!")

    ## Load Dataset
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(args.eval_data_file, "r") as file:
        eval_data = json.load(file)

    source_codes = []
    with open(args.eval_data_file) as f:
        datas = json.load(f)
        for data in datas:
            code = data['input']
            source_codes.append(code)

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code))

    success_attack = 0
    total_cnt = 0

    recoder = Recorder_style(args.csv_store_path)
    attacker = DIP_Attacker(args, 0, model, tokenizer)
    query_times = 0
    code_lines = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code))
        lines = code.split("\n")
        for line in lines:
            if len(line.split(" ")) > 5:
                code_lines.append(line.strip())

    start_time = time.time()
    tokenizer_dip = RobertaTokenizer.from_pretrained('../CodeBERT/codebert-base')
    model_dip = RobertaModel.from_pretrained('../CodeBERT/codebert-base', output_attentions=True)
    model_dip.to('cuda')

    for index, example in enumerate(eval_data):
        if 0 <= index < 600:  # or index<=600:
            code = example["input"]
            label = example["output"]
            orig_prob, orig_pred, flag = get_llm_result(code, model, tokenizer, label)
            if flag == -1:
                recoder.write(index, None, None, None, None, None, None, "0")
                continue

            print_index(index, label, orig_pred)

            if orig_pred != label:
                recoder.write(index, None, None, None, None, None, None, "0")
                continue
            total_cnt += 1
            example_start_time = time.time()

            code_list = get_dip_sub(code)
            dis_codes = random.sample(source_codes, 10)
            dis_code_dist = dip_comp_code(code, dis_codes, False, tokenizer_dip, model_dip)
            dis_codes = list(dis_code_dist.keys())
            dis_snippets = []
            for idx, dis_code in enumerate(dis_codes):
                dis_snippets.append(get_dip_snippet(dis_code, tokenizer_dip, model_dip))
            lines_dist = dip_comp_code(code, code_list, True, tokenizer_dip, model_dip)
            lines = list(lines_dist.keys())

            # attack begin
            is_success, adv_code, query = attacker.dip_attack(code, dis_snippets, lines, label)

            example_end_time = (time.time() - example_start_time) / 60
            print("Example time cost: ", round(example_end_time, 2), "min")
            print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
            print("Query times in this attack: ", query)

            if is_success == 1:
                prob, adv_pred, flag = get_llm_result(adv_code, model, tokenizer, label)
                if adv_pred != label or flag == -1:
                    print("true adv!")
                    success_attack += 1
                    recoder.write(index, code, adv_code, label, adv_pred, query_times, round(example_end_time, 2),
                                  "DIP")

            else:
                recoder.write(index, None, None, None,
                              None, query, example_end_time, "0")
            print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == "__main__":
    main()