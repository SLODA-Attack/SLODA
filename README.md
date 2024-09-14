# SLODA

This is the codebase for the paper "Statement-level Adversarial Attack on Vulnerability Detection Models via Out-Of-Distribution Features".

## Attack Approach

- [MHM](https://github.com/SEKE-Adversary/MHM)
- DIP
- [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/)
- [CODA](https://github.com/tianzhaotju/CODA/tree/main)


## Experiments


### Create Environment


you need to install [srcml](https://www.srcml.org/) and [tree_sitter](https://tree-sitter.github.io/tree-sitter/) to run the code.

### Fine-tuning CodePTMs

Use `train.py` to train models.

Take an example:

```
cd CodeBERT/code
python train.py
```

### Fine-tuning CodeLlama

Install LLaMA-Factory:

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
In the LLaMA-Factory, please modify the `dataset_info.json` in the `data` path according to your actual path, and change `do_sample` to `False` in `protocol.py` under `src/llamafactory/api` to avoid randomness in LLM output affecting the adversarial attacks.

Then, use the following command to run LoRA fine-tuning of the `CodeLlama-7B` model on Devign dataset.

```
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default  \
    --flash_attn auto \
    --dataset_dir data \
    --dataset Devign \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir CodeLlama-Devign \
    --overwrite_output_dir True  \
    --fp16 True \
    --plot_loss True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --do_eval True  \
    --eval_steps 100 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --load_best_model_at_end
```

### Attacking CodePTMs

In our study, we employed three balanced datasets: Devign, BigVul and DiverseVul. The dataset is in ./Adv-attack/dataset.


Take an example, when using MHM to attack Devign on CodeBERT, the script example is as follows.

```
CUDA_VISIBLE_DEVICES=0 python attack_ptm_mhm.py \
    --model_type roberta \
    --output_dir ../../CodeBERT/saved_models_Devign/ \
    --tokenizer_name microsoft/codebert-base \
    --model_name_or_path microsoft/codebert-base \
    --csv_store_path attack_results_ptm/attack_mhm_CodeBERT_Devign.csv \
    --eval_data_file ../../dataset/Devign/test.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 12345 2>&1 | tee attack_results_ptm/attack_mhm_CodeBERT_Devign.log
```

The result will be saved in `csv_store_path`.  
Run experiments of GraphCodeBERT and UniXcoder as well.  
`./CodeBERT/` contains code for the CodeBERT experiment and `./GraphCodeBERT` contains code for GraphCodeBERT experiment and  `./UniXcoder ` contains code for UniXcoder experiment.


**Running CODA**

Use `attack_ptm_coda.py` to run CODA.  
```
CUDA_VISIBLE_DEVICES=0 python attack_ptm_coda.py \
    --eval_data_file ../dataset/Devign/test.jsonl \
    --model_name microsoft/codebert-base \
    --csv_store_path attack_results_ptm/attack_CODA_CodeBERT_Devign.csv \
    2>&1 | tee attack_results_ptm/attack_CODA_CodeBERT_Devign.log
```

### Attacking CodeLlama 

Use `attack_llm_mhm.py` to run CodeLlama attack.
The content of `attack_llm_mhm.py` is similar to `attack_ptm_mhm.py`.

### Adversarial Snippets 

The adversarial snippets are stored in `./features`.The code of getting-snippet is in `.get_snippet/get_snippet.py`. 

## Acknowledgement

We are very grateful that the authors of CodeBERT, GraphCodeBERT, UniXcoder, MHM , CODA, ALERT, srcml and tree_sitter make their code publicly available so that we can build this repository on top of their code. 
