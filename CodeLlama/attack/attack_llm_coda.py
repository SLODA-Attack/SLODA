import sys
import os

sys.path.append('../../dataset/')
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import warnings
import time
from attack_utils import build_vocab, set_seed
from attack_utils import Recorder, get_llm_result,Recorder_style
from llm_attacker import CODA_Attacker

import torch
import os
import sys
import json
import argparse
import fasttext
from peft import PeftModel

from transformers import LlamaTokenizer, CodeLlamaTokenizer, LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM

from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaForSequenceClassification,
                          RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)

def print_index(index, idx, label, orig_pred):
    entry = f"Index: {index} | Idx: {idx} | label: {label} | orig_pred: {orig_pred}"
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
    print(f"Base model loaded from {args.base_model}!")

    model = PeftModel.from_pretrained(model, args.tuned_model)
    print(f"Tuned model loaded from {args.tuned_model}!")

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

    fasttext_model = fasttext.load_model("../dataset/fasttext/fasttext.bin")
    codebert_mlm = RobertaForMaskedLM.from_pretrained("../CodeBERT/codebert-base")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("../CodeBERT/codebert-base")
    codebert_mlm.to('cuda')
    generated_substitutions = json.load(open('./codebert_all_subs_Devign.json', 'r', encoding='utf-8'))
    attacker = CODA_Attacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, fasttext_model,
                             generated_substitutions)
    recoder = Recorder_style(args.csv_store_path)
    start_time = time.time()
    success_attack = 0
    total_cnt = 0

    for index, example in enumerate(eval_data):
        if 992 <= index < 1000:

            print("index", index)
            code = source_codes[index]

            if index >= len(source_codes):
                break
            query_times = 0
            example_start_time = time.time()

            is_success, final_code, min_gap_prob, query = attacker.attack(
                example,
                code,
                query_times
            )
            trans_end_time = (time.time() - example_start_time) / 60
            print("Example time cost: ", round(trans_end_time, 2), "min")
            if is_success >= -1:
                total_cnt += 1
                if is_success >= 1:
                    success_attack += 1
                    recoder.write(index, code, final_code, None, None, query, None,"CODA")
                if total_cnt == 0:
                    continue
                print("Success rate: %.2f%%" % ((1.0 * success_attack / total_cnt) * 100))
                print("Successful items count: ", success_attack)
                print("Total count: ", total_cnt)
                print("Index: ", index)
                print()

    print(len(eval_data), flush=True)
    

if __name__ == "__main__":
    main()