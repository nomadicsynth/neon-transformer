from dataclasses import dataclass, field
from typing import Dict

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from neon import NeonConfig, NeonForCausalLM


@dataclass
class ModelArguments:
    model_size: str = field(
        default="spark",
        metadata={
            "help": "Size variant of the model to train (spark, glow, beam, arc, nova)"
        },
    )


@dataclass
class DataArguments:
    max_seq_length: int = field(default=512)
    dataset_name: str = field(default="wikimedia/wikipedia")
    dataset_config_name: str = field(default="20231101.simple")
    num_train_samples: int = field(default=1000)
    num_eval_samples: int = field(default=100)
    streaming: bool = field(default=False)


def get_model_config(model_size: str) -> NeonConfig:
    """Get model configuration based on size variant."""
    configs = {
        "spark": dict(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            head_dim=64,
        ),
        "glow": dict(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            intermediate_size=3072,
            head_dim=64,
        ),
        "beam": dict(
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=4096,
            head_dim=64,
        ),
        "arc": dict(
            hidden_size=1536,
            num_hidden_layers=24,
            num_attention_heads=20,
            num_key_value_heads=20,
            intermediate_size=6144,
            head_dim=64,
        ),
        "nova": dict(
            hidden_size=2048,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=24,
            intermediate_size=8192,
            head_dim=64,
        ),
    }

    if model_size not in configs:
        raise ValueError(
            f"Model size {model_size} not supported. Choose from {list(configs.keys())}"
        )

    base_config = dict(
        max_position_embeddings=2048,
        vocab_size=32000,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        activation_function="silu",
        initializer_range=0.02,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    config_dict = {**base_config, **configs[model_size]}
    return NeonConfig(**config_dict)


def prepare_dataset(args: DataArguments):
    """Load and prepare the dataset with support for both regular and streaming modes."""
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized_examples = {}
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            # return_tensors="pt",
        )

        # Add labels to the tokenized examples
        tokenized_examples["labels"] = tokenized_examples["input_ids"]

        return tokenized_examples

    # Load dataset with streaming if specified
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        streaming=args.streaming,
        split="train",
    )

    if args.streaming:
        # For streaming datasets, use take() and skip()
        train_dataset = dataset.take(args.num_train_samples)
        # Skip the training samples to get to the validation set
        temp_dataset = dataset.skip(args.num_train_samples)
        validation_dataset = temp_dataset.take(args.num_eval_samples)

        from datasets import IterableDatasetDict

        dataset = IterableDatasetDict()
        dataset["train"] = train_dataset
        dataset["test"] = validation_dataset

        # Apply tokenization to streaming datasets
        dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset["train"].column_names,
        )
    else:
        dataset = dataset["train"].train_test_split(
            test_size=args.num_eval_samples,
            train_size=args.num_train_samples,
            shuffle=True,
            seed=42,
        )

        # Apply tokenization to regular datasets
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4,
        )

    return dataset, tokenizer


def main():
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare dataset
    datasets, tokenizer = prepare_dataset(data_args)
    if data_args.streaming:
        training_args.max_steps = (
            data_args.num_train_samples // training_args.per_device_train_batch_size
        )

    # Initialize model
    config = get_model_config(model_args.model_size)
    # config._attn_implementation = "eager"
    config._attn_implementation = "flash_attention_2"
    # config._attn_implementation = "sdpa"
    model = NeonForCausalLM(config)

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()


if __name__ == "__main__":
    # Example usage:
    # python train_neon.py \
    #     --model_size spark \
    #     --output_dir ./neon-test \
    #     --num_train_epochs 3 \
    #     --per_device_train_batch_size 4 \
    #     --per_device_eval_batch_size 4 \
    #     --logging_steps 10 \
    #     --save_steps 50 \
    #     --evaluation_strategy steps \
    #     --eval_steps 50 \
    #     --save_total_limit 2 \
    #     --load_best_model_at_end True \
    #     --num_train_samples 1000 \
    #     --num_eval_samples 100

    main()
