import os

os.system("CUDA_VISIBLE_DEVICES=1 python get_substitutes_vulnerability.py \
    --store_path test_subs_vulnerability_DiverseVul_part_4_2.jsonl \
    --base_model=codebert-base-mlm \
    --eval_data_file=../DiverseVul/test_sub_part_4.jsonl \
    --block_size 512")