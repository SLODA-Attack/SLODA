import os

model_path = "../graphcodebert-base"


data_dir = "../../dataset/DiverseVul"#"../../dataset/BigVul""../../dataset/Devign""../../dataset/ReVeal""../../dataset/DiverseVul"
output_dir = "../saved_models_DiverseVul"
command = f"""
CUDA_VISIBLE_DEVICES=2 python run.py \
    --output_dir={output_dir} \
    --config_name={model_path} \
    --model_name_or_path={model_path} \
    --tokenizer_name={model_path} \
    --do_test \
    --train_data_file={data_dir}/train.jsonl \
    --eval_data_file={data_dir}/valid.jsonl \
    --test_data_file={data_dir}/test.jsonl \
    --block_size 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --csv_path {output_dir}/test.csv \
    --seed 12345 2>&1| tee {output_dir}/test.log
"""
os.system(command)