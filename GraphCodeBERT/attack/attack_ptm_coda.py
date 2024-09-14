
import fasttext
import copy
import sys
import os

import numpy as np

sys.path.append('../../dataset/')
sys.path.append('../code/')   # CodeBERT, UniXcoder, GraphCodeBERT, EPVD, ReGVD

from run import convert_examples_to_features, TextDataset, InputFeatures

from python_parser.parser_folder import remove_comments_and_docstrings
import json

import argparse
import warnings
import torch
import time

from attack_utils import set_seed

from attack_utils import Recorder_style
from model import Model
from ptm_attacker import CODA_Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaForSequenceClassification,
                          RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    
}

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

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument('--head', type=int,
                        help="number of dataset examples to cut off")
    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.seed = 123456
    args.number_labels = 2
    args.eval_batch_size = 64
    args.language_type = 'c'
    args.n_gpu = 2
    args.block_size = 512
    args.use_ga = True

    if args.model_name == 'codebert':
        args.output_dir = '../CodeBERT/saved_models_DiverseVul'
        args.model_type = 'codebert_roberta'
        args.config_name = '../CodeBERT/codebert-base'
        args.model_name_or_path = '../CodeBERT/codebert-base'
        args.tokenizer_name = '../CodeBERT/codebert-base'
        args.base_model = '../CodeBERT/codebert-base'
    if args.model_name == 'graphcodebert':
        args.output_dir = '../GraphCodeBERT/saved_models_DiverseVul'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = '../GraphCodeBERT/graphcodebert-base'
        args.tokenizer_name = '../GraphCodeBERT/graphcodebert-base'
        args.model_name_or_path = '../GraphCodeBERT/graphcodebert-base'
        args.base_model = '../GraphCodeBERT/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 128
    if args.model_name == 'unixcoder':
        args.output_dir = '../UniXcoder/saved_models_DiverseVul'
        args.model_type = 'unixcoder'
        args.config_name = '../UniXcoder/unixcoder-base'
        args.model_name_or_path = '../UniXcoder/unixcoder-base'
        args.tokenizer_name = '../UniXcoder/unixcoder-base'
        args.base_model = '../UniXcoder/unixcoder-base'
    set_seed(args.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES['graphcodebert_roberta']
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_name_or_path == 'codebert':
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, 510)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    if args.model_name == 'codebert':
        model = Model(model, config,  args)
    elif args.model_name == 'graphcodebert':
        model = Model(model, config,  args)
    elif args.model_name == 'unixcoder':
        model = Model(model, config,  args)
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))      
    model.to(args.device)

    if args.model_name == 'codebert':
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'unixcoder':
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    fasttext_model = fasttext.load_model("../../dataset/fasttext/fasttext_DiverseVul.bin")
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')
    generated_substitutions = json.load(open('../../dataset/coda_preprocess/codebert_all_subs_Devign.json', 'r', encoding='utf-8'))
    attacker = CODA_Attacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions)
    
    
    recoder = Recorder_style(args.csv_store_path)
    
    
    source_codes=[]
    label = []
    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            code = js['func']
            source_codes.append(code)

    assert (len(source_codes) == len(eval_dataset))
    
    success_attack = 0
    total_cnt = 0

    for index, example in enumerate(eval_dataset):
        print("index",index)
        code = source_codes[index]
        code = remove_comments_and_docstrings(code,"c")
        tokenized_code_tokens = tokenizer_mlm.tokenize(code)
        if len(tokenized_code_tokens)>512:

            continue
        if index >= len(source_codes):
            break
        query_times=0
        
        is_success, final_code, min_gap_prob,query = attacker.attack(
            example,
            code,
            query_times
        )
        
        if is_success >= -1:
            total_cnt += 1
            if is_success >= 1:
                success_attack += 1
                recoder.write(index, code, final_code, None, None, query, None)
            if total_cnt == 0:
                continue
            print("Success rate: %.2f%%" % ((1.0 * success_attack / total_cnt) * 100))
            print("Successful items count: ", success_attack)
            print("Total count: ", total_cnt)
            print("Index: ", index)
            print()


if __name__ == '__main__':
    main()

