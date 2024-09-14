import os
import json
import sys
import argparse
from tqdm import tqdm
import sys
import os
import sys
sys.setrecursionlimit(3000)  
sys.path.append('../dataset/')
sys.path.append('../CodeBERT/code/')# CodeBERT, UniXcoder, GraphCodeBERT, EPVD, ReGVD
sys.path.append('../dataset/python_parser')
from python_parser.run_parser_CODA import get_identifiers, remove_comments_and_docstrings, get_example_batch
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification)
from model import Model
from run import convert_examples_to_features, TextDataset, InputFeatures, convert_code_to_features
import torch
import numpy as np
import copy


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
    return words, sub_words, keys


MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "c")
    code_tokens = get_identifiers(remove_comments_and_docstrings(new_code, "c"), "c", True)
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--all_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    
    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.seed = 123456
    args.block_size = 512
    args.eval_batch_size = 32
    args.number_labels = 1
    args.language_type = 'c'
    args.store_path = './%s_all_subs_BigVul.json' % args.model_name

    if args.model_name == 'codebert':
        args.output_dir = '../CodeBERT/saved_models_BigVul'
        args.model_type = 'codebert_roberta'
        args.config_name = '../CodeBERT/codebert-base'
        args.model_name_or_path = '../CodeBERT/codebert-base'
        args.tokenizer_name = '../CodeBERT/codebert-base'
        args.base_model = '../dataset/proprocess/codebert-base-mlm'
    if args.model_name == 'graphcodebert':
        args.output_dir = '../GraphCodeBERT/saved_models_BigVul'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = '../GraphCodeBERT/graphcodebert-base'
        args.tokenizer_name = '../GraphCodeBERT/graphcodebert-base'
        args.model_name_or_path = '../GraphCodeBERT/graphcodebert-base'
        args.base_model = '../GraphCodeBERT/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)
        
    model = Model(model, config,  args)
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    all_dataset = TextDataset(tokenizer, args, args.all_data_file)

    source_codes = []
    with open(args.all_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            code = js['func']
            source_codes.append(code)
    assert (len(source_codes) == len(all_dataset))
    print('length of all data', len(source_codes))

    all_labels = {}
    count = 0
    with open(args.store_path, "w") as wf:
        for index, example in tqdm(enumerate(all_dataset)):
            code = source_codes[index]
            code = remove_comments_and_docstrings(code, "c")
            tokenized_code_tokens = tokenizer_mlm.tokenize(code)
            if len(tokenized_code_tokens)>512:
                continue
            
            logits, preds = model.get_results([example], args.eval_batch_size)
            if args.model_name == 'codebert':
                true_label = str(int(example[1].item()))
            elif args.model_name == 'graphcodebert':
                true_label = str(int(example[3].item()))

            orig_prob = np.max(logits[0])
            orig_label = str(int(preds[0]))
            

            if not true_label == orig_label:
                continue

            if true_label not in all_labels.keys():
                all_labels[true_label] = []

            try:
                variable_name, function_name, _ = get_identifiers(remove_comments_and_docstrings(code, "c"), "c")
            except:
                variable_name, function_name, _ = get_identifiers(code, "c")

            variables = []
            variables.extend(variable_name)
            variables.extend(function_name)

            embeddings = get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args)

            if not os.path.exists('./%s_all_subs_BigVul' % args.model_name):
                os.makedirs('./%s_all_subs_BigVul' % args.model_name)
            np.save('./%s_all_subs_BigVul/%s_%s' % (args.model_name, str(orig_label), str(index)), embeddings.cpu().numpy())
            all_labels[true_label].append({'code': code, 'embeddings_index': index, 'variable_name': variable_name, 'function_name': function_name})
            count += 1
        print(count, len(all_dataset), count/len(all_dataset))
        wf.write(json.dumps(all_labels) + '\n')


if __name__ == "__main__":
    main()

