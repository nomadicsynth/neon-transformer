#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_size>"
    echo "Example: $0 4s"
    echo "Valid model sizes: 2s, 4s, 8s"
    exit 1
fi

model_size=$1

# Pull NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:23.12-py3

# System-level setup that persists outside container
apt update && apt install -y python3-pip git python3-venv

# Clone repo (do this outside container)
git clone https://github.com/nomadicsynth/neon-transformer.git
cd neon-transformer

echo "Starting training for model size: ${model_size}"

# Run training in container
docker run --gpus all -v $(pwd):/workspace -w /workspace --rm nvcr.io/nvidia/pytorch:23.12-py3 bash -c "
    # Container-level setup
    # Create a venv
    # python3 -m venv .venv
    # source .venv/bin/activate

    PIP_REQUIRE_VIRTUALENV=false
    
    pip install -r requirements-env.txt
    pip install -r requirements-app.txt
    pip install flash-attn --no-build-isolation

    # Set up wandb and huggingface
    wandb login
    huggingface-cli login

    # Run training
    python train_mistral_baseline.py \
        --output_dir './outputs/mistral-${model_size}-baseline' --overwrite_output_dir True \
        --bf16 True --bf16_full_eval True \
        --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.98 \
        --learning_rate 3.2e-4 --lr_scheduler_type cosine --weight_decay 0.01 \
        --max_steps 10000 --warmup_steps 500 \
        --logging_strategy steps --logging_steps 100 \
        --report_to wandb --watch 'all' --wandb_log_model 'end' \
        --project_name neon-test --run_name 'mistral-${model_size}-baseline' \
        --eval_strategy steps --eval_steps 1000 --eval_on_start True \
        --save_strategy steps --save_steps 1000 --save_total_limit 10 --load_best_model_at_end True \
        --dataset_name HuggingFaceFW/fineweb --dataset_config_name sample-10BT \
        --num_train_samples 0 --num_eval_samples 5000 \
        --streaming True --packing True --dataloader_num_workers 2 \
        --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
        --max_seq_length 2048 --gradient_accumulation_steps 8 \
        --seed 42 --model_size '${model_size}'
"

echo "Completed training for model size: ${model_size}"