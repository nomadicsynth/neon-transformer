#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <wandb_sweep_id> <wandb_key> <hf_key>"
    exit 1
fi

sweep_id=$1
wandb_key=$2
hf_key=$3

# Pull NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.10-py3

# System-level setup that persists outside container
sudo apt update && sudo apt install -y python3-pip git nvtop

# Clone repo (do this outside container)
git clone https://github.com/nomadicsynth/neon-transformer.git
cd neon-transformer

echo "Starting W&B Agent for Sweep ${sweep_id}"

# Run Agent in container with API keys passed in
docker run --runtime=nvidia \
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
    conda install -y pytorch pytorch-cuda=12.4 flash-attn==2.6.3 -c pytorch -c nvidia -c conda-forge

    # Container-level setup
    pip install -r requirements-env.txt
    pip install -r requirements-app.txt

    # Run Agent
    wandb agent neon-cortex/neon-test/${sweep_id}
"

echo "Agent Shutdown"
