#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_size> <wandb_key> <hf_key>"
    echo "Example: $0 4s your-wandb-key your-hf-key"
    echo "Valid model sizes: 2s, 4s, 8s"
    exit 1
fi

model_size=$1
wandb_key=$2
hf_key=$3

# Pull NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.10-py3

# System-level setup that persists outside container
apt update && apt install -y python3-pip git nvtop

# Clone repo (do this outside container)
git clone https://github.com/nomadicsynth/neon-transformer.git
cd neon-transformer

echo "Starting training for model size: ${model_size}"

# Run training in container with API keys passed in
docker run --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    -e WANDB_API_KEY="${wandb_key}" \
    -e HUGGING_FACE_HUB_TOKEN="${hf_key}" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm nvcr.io/nvidia/pytorch:24.10-py3 bash -c "
    # Install conda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/conda
    eval \"\$(/opt/conda/bin/conda shell.bash hook)\"
    
    # Create and activate conda environment
    conda create -n neon python=3.12 -y
    conda activate neon
    
    # Install pytorch with CUDA
    conda install -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
    
    # Install flash-attention
    conda install -y flash-attn>=2.6.0 -c conda-forge

    # Container-level setup
    pip install -r requirements-env.txt
    pip install -r requirements-app.txt

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