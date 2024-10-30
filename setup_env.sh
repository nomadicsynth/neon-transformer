#!/bin/bash
# System stuff
apt update && apt install -y python3-pip git python3.12-venv

# Create a venv
python3 -m venv .venv
source .venv/bin/activate

# Clone your repo
git clone https://github.com/nomadicsynth/neon-transformer.git
cd neon-transformer

# Set up Python environment
python3 -m pip install -r requirements-env.txt
python3 -m pip install -r requirements-app.txt

# Set up wandb and huggingface
wandb login
huggingface-cli login

# Run!
model_sizes=("2s", "4s", "8s")
for model_size in "${model_sizes[@]}"; do
    python train_mistral_baseline.py \
                    --output_dir "./outputs/mistral-${model_size}-baseline" --overwrite_output_dir True \
                    --bf16 True --bf16_full_eval True \
                    --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
                    --learning_rate 3.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
                    --max_steps 10000 --warmup_steps 500 \
                    --logging_strategy steps --logging_steps 100 \
                    --report_to wandb --watch "all" --wandb_log_model "end" \
                    --project_name neon-test --run_name "mistral-${model_size}-baseline" \
                    --eval_strategy steps --eval_steps 1000 --eval_on_start True \
                    --save_strategy steps --save_steps 1000 --save_total_limit 10 --load_best_model_at_end True \
                    --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
                    --num_train_samples 0 --num_eval_samples 5000 \
                    --streaming True --packing True --dataloader_num_workers 2 \
                    --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
                    --max_seq_length 2048 --gradient_accumulation_steps 8 \
                    --seed 42 --model_size "${model_size}"
done
