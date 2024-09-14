import copy
import sys
import os

import numpy as np

sys.path.append('../../dataset/')
sys.path.append('../code/')   # CodeBERT, UniXcoder, GraphCodeBERT, EPVD, ReGVD
from attack_utils import CodeDataset, GraphCodeDataset, EPVDDataset
from run import convert_examples_to_features, TextDataset, InputFeatures
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from data_augmentation import get_identifiers, get_code_tokens
from python_parser.parser_folder import remove_comments_and_docstrings
import json
import logging
import argparse
import warnings
import torch
import time
import random
from attack_utils import build_vocab, set_seed
from model import Model
from attack_utils import Recorder_style
from ptm_attacker import ALERT_Attacker, DIP_Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaForSequenceClassification,
                          RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'epvd': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


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
        if index == 0 or index == len(code_list) - 1 or index == 1 or index == 2 or index == 3:
            continue
        new_code_list = copy.deepcopy(code_list)
        new_code_list.insert(index, unk_line)
        new_code = "\n".join(new_code_list)
        ans_code_list.append(new_code)
    return ans_code_list


def get_dip_snippet(code,tokenizer,model):
    lines = code.split('\n')

    # 对输入代码进行分词
    inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512)

    # 将tokens传入模型以获取注意力得分
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


def dip_comp_code(code1, code_list, line_flag,tokenizer,model):
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
            with torch.no_grad():
                outputs_original = model(**inputs_original)
                outputs_modified = model(**inputs_modified)
            cls_representation_original = outputs_original.last_hidden_state[:, 0, :].detach().numpy().flatten()
            cls_representation_modified = outputs_modified.last_hidden_state[:, 0, :].detach().numpy().flatten()
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
            with torch.no_grad():
                outputs_original = model(**inputs_original)
                outputs_modified = model(**inputs_modified)
            cls_representation_original = outputs_original.last_hidden_state[:, 0, :].detach().numpy().flatten()
            cls_representation_modified = outputs_modified.last_hidden_state[:, 0, :].detach().numpy().flatten()
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
    print(result)




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

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
    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--cnn_size', type=int, default=128, help="For cnn size.")
    parser.add_argument('--filter_size', type=int, default=3, help="For cnn filter size.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="hidden size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    parser.add_argument("--index", nargs='+',type=list, help="Optional input sequence length after tokenization.")
    args = parser.parse_args()
    print(args.index)
    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)
    ptm_model = args.output_dir.split("/")[-3]
    args.start_epoch = 0
    args.start_step = 0
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1 if ptm_model in {"CodeBERT", "ReGVD"} else 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model =  Model(model, config, args)
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    print(output_dir)
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    print(f"Loaded tuned model from {output_dir}!")

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # Load original source codes
    source_codes = []

    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            code = js['func']
            source_codes.append(code)

    assert (len(source_codes) == len(eval_dataset))

    success_attack = 0
    total_cnt = 0

    recoder = Recorder_style(args.csv_store_path)
    query_times = 0
    
    tokenizer_dip = RobertaTokenizer.from_pretrained('../CodeBERT/codebert-base')
    model_dip = RobertaModel.from_pretrained('../CodeBERT/codebert-base', output_attentions=True)
    attacker = DIP_Attacker(args, ptm_model,model, tokenizer, tokenizer_dip)
    start_time = time.time()

    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        code = remove_comments_and_docstrings(code, "c")
        orig_prob, orig_pred = model.get_results([example], args.eval_batch_size)
        tokenized_code_tokens = tokenizer.tokenize(code)
        orig_pred = orig_pred[0]
        label = example[3].item() if ptm_model == "GraphCodeBERT" else example[1].item()
        print_index(index, label, orig_pred)
        if len(tokenized_code_tokens) > 512:
            continue
        if orig_pred != label:
            recoder.write(index, None, None, None, None, None, None)
            continue
        total_cnt += 1

        example_start_time = time.time()
        code_list = get_dip_sub(code)
        dis_codes = random.sample(source_codes, 30)
        dis_code_dist = dip_comp_code(code, dis_codes, False, tokenizer_dip, model_dip)
        dis_codes = list(dis_code_dist.keys())
        dis_snippets = []
        for idx, dis_code in enumerate(dis_codes):
            dis_snippets.append(get_dip_snippet(dis_code, tokenizer_dip, model_dip))
        lines_dist = dip_comp_code(code, code_list, True, tokenizer_dip, model_dip)
        lines = list(lines_dist.keys())

        # attack begin
        is_success, adv_code = attacker.dip_attack(example, code, dis_snippets, lines, label)
        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)


        if is_success == 1:
            success_attack += 1
            recoder.write(index, code, adv_code, None,
                          None, model.query - query_times, example_end_time)

        else:
            recoder.write(index, None, None, None,
                          None, model.query - query_times, example_end_time)
        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

    if __name__ == "__main__":
        main()
