import sys
import os

sys.path.append('../../dataset/')
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import warnings
import time
from attack_utils import build_vocab, set_seed
from attack_utils import Recorder, get_llm_result
from llm_attacker import ALERT_Attacker
import torch
import os
import sys
import json

import argparse
from peft import PeftModel
from transformers import GenerationConfig
from transformers import LlamaTokenizer, CodeLlamaTokenizer, LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)

def print_index(index, label, orig_pred):
    entry = f"Index: {index} | label: {label} | orig_pred: {orig_pred}"
    entry_length = len(entry)
    padding = 100 - entry_length
    left_padding = padding // 2
    right_padding = padding - left_padding
    result = f"{'*' * left_padding} {entry} {'*' * right_padding}"
    print(result)

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

    # source_codes = []
    # with open(args.eval_data_file) as f:
    #     datas = json.load(f)
    #     for data in datas:
    #         code = data['input']
    #         source_codes.append(code)

    # Load original source codes
    source_codes = []
    generated_substitutions = []
    idxs = []

    dataset = args.eval_data_file.split("/")[-2]
    subs_data_file = f"../dataset/proprocess/test_subs_vulnerability_{dataset}.jsonl"
    with open(subs_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            code = js['func']
            source_codes.append(code)
            generated_substitutions.append(js['substitutes'])
            idxs.append(js['idx'])
    print(len(source_codes), len(eval_data), len(generated_substitutions))
    # assert (len(source_codes) == len(eval_data) == len(generated_substitutions))

    success_attack = 0
    total_cnt = 0

    recoder = Recorder(args.csv_store_path)
    attacker = ALERT_Attacker(args, model, tokenizer, use_bpe=1, threshold_pred_score=0)

    start_time = time.time()

    for index, example in enumerate(eval_data):
        if 543 <= index < 600:
            code = example["input"]
            label = example["output"]
            idx_llm = example["idx"]
            orig_code_tokens = get_code_tokens(code)
            identifiers = get_identifiers(code)
            orig_prob, orig_pred, flag = get_llm_result(code, model, tokenizer, label)
            print_index(index, label, orig_pred)
            if flag == -1:
                recoder.write(index, None, None, None, None, None, None, None, "0")
                continue
            if orig_pred != label:
                recoder.write(index, None, None, None, None, None, None, None, "0")
                continue

            substitutes = None
            for idx, substitute in zip(idxs, generated_substitutions):
                if idx == idx_llm:
                    substitutes = substitute
            if len(substitutes) == 0:
                recoder.write(index, None, None, None, None, None, None, None, "0")
                continue

            total_cnt += 1
            print(substitutes)
            example_start_time = time.time()
            print("Start Greedy!")
            code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(
                example, code, label, orig_prob, substitutes)
            attack_type = "Greedy"
            print("Start GA!")
            if is_success == -1:
                code, prog_length, adv_code, query, true_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
                    code, query, substitutes, label, initial_replace=replaced_words)
                attack_type = "GA"

            example_end_time = (time.time() - example_start_time) / 60
            print("Example time cost: ", round(example_end_time, 2), "min")
            print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
            print("Query times in this attack: ", query)
            replace_info = ''
            if replaced_words is not None:
                for key in replaced_words.keys():
                    replace_info += key + ':' + replaced_words[key] + ','

            if is_success == 1:
                prob, adv_pred, flag = get_llm_result(adv_code, model, tokenizer, label)
                if adv_pred != label or flag == -1:
                    print("true adv!")
                    success_attack += 1
                    recoder.write(index, code, adv_code, len(orig_code_tokens), len(identifiers),
                                  replace_info, query, example_end_time, attack_type)

            else:
                recoder.write(index, None, None, len(orig_code_tokens), len(identifiers),
                              None, query, example_end_time, "0")
            print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

if __name__ == "__main__":
    main()