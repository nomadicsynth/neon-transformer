# Neon

Neon is an experimental language model architecture that combines the strengths of transformer-based models with a novel function mechanism called FlamingMoE. The goal of Neon is to create a more flexible and efficient language model that can learn to adapt to a wide range of tasks and domains.

## Architecture

Neon is based on the Mistral architecture, with additional components including:

* DiffAttention: a differential attention mechanism that reduces noise and improves performance
* nGPT: a normalization scheme that keeps the hidden states on a hypersphere, reducing hallucinations and token requirements for training

## FlamingMoE

FlamingMoE is an experimental function mechanism that allows the model to selectively apply different functions to the input data during each step of the processing pipeline. This mechanism is inspired by the concept of mixture of experts, but with a more flexible and dynamic approach. Unlike traditional function-calling mechanisms, which operate in-context and can only be invoked at specific points in the code, FlamingMoE works within the model's internal processing pipeline, allowing the model to apply different functions to the input data at each step. The current implementation of FlamingMoE uses a bank of learned functions, which can be combined and applied to the input data in a flexible and dynamic way. However, the long-term vision for FlamingMoE is to expand it into a general-purpose input/output (IO) mechanism for the model, allowing it to interact with external resources, such as databases, APIs, or even other models, during the processing pipeline. This would enable the model to access and incorporate a wide range of external information and knowledge, and to generate more informed and accurate outputs. While FlamingMoE is still an unproven experiment, it has the potential to significantly enhance the capabilities and flexibility of the Neon model, and to open up new possibilities for natural language processing and generation.

## Warning

FlamingMoE is an experimental and WIP feature and should be considered unstable. It may not work as intended, and may require significant tuning and debugging to achieve good results. Use at your own risk!

## Getting Started

To get started with Neon, please follow these steps:

1. Install the required dependencies, including PyTorch and the Transformers library
2. Clone the repository and navigate to the root directory
3. Run the `train.py` script to train the model on your dataset of choice

## Weights and Biases Agent

To run W&B Agent jobs on cloud compute:
```
wget https://raw.githubusercontent.com/nomadicsynth/neon-transformer/refs/heads/main/wandb_agent.sh -O wandb_agent.sh
bash wandb_agent.sh <wandb_sweep_id> <wandb_key> <hf_key>
```

## Contributing

We welcome contributions to the Neon project, including bug fixes, new features, and improvements to the FlamingMoE mechanism. Please submit pull requests and issues through the GitHub repository.

## License

Neon is released under the Apache 2.0 license. See the `LICENSE` file for more information.
