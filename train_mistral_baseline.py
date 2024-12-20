print("Loading imports")
from dataclasses import dataclass, field

import evaluate
import torch
from datasets import load_dataset
from transformers import EvalPrediction
from trl import SFTConfig, SFTTrainer

from transformers import MistralConfig, MistralForCausalLM


@dataclass
class ModelArguments:
    model_size: str = field(
        default="2s",
        metadata={
            "help": "Size variant of the model to train (2s, 4s, 8s)",
            "choices": ["2s", "4s", "8s"],
        },
    )


@dataclass
class DataArguments:
    dataset_name: str = field(default="HuggingFaceFW/fineweb", metadata={"help": "Name of the dataset to use"})
    dataset_config_name: str = field(default="sample-10BT", metadata={"help": "Name of the dataset configuration"})
    num_train_samples: int = field(default=1000, metadata={"help": "Number of training samples"})
    num_eval_samples: int = field(default=100, metadata={"help": "Number of evaluation samples"})
    streaming: bool = field(default=True, metadata={"help": "Whether to use streaming mode"})


@dataclass
class WandbArguments:
    project_name: str = field(default="neon-test", metadata={"help": "Name of the W&B project"})
    watch: str = field(default="false", metadata={"help": "Whether to watch the training", "choices": ["all", "gradients", "parameters", "false"]})
    wandb_log_model: str = field(default="false", metadata={"help": "Whether to log the model", "choices": ["end", "checkpoint", "false"]})


def get_model_config(args: ModelArguments) -> MistralConfig:
    """Get model configuration based on size variant."""

    # `intermediate_size` should be 8/3 * `hidden_size`
    configs = {
        "2s": dict(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=1360,
        ),
        "4s": dict(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=1360,
        ),
        "8s": dict(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=1360,
        ),
    }

    if args.model_size not in configs:
        raise ValueError(
            f"Model size {args.model_size} not supported. Choose from {list(configs.keys())}"
        )

    base_config = dict(
        max_position_embeddings=2048,
        vocab_size=32000,
        attention_dropout=0.0,
        hidden_dropout=0.1,
        activation_function="silu",
        initializer_range=0.02,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    config_dict = {**base_config, **configs[args.model_size]}
    return MistralConfig(**config_dict)


def prepare_dataset(args: DataArguments):
    """Load and prepare the dataset with support for both regular and streaming modes."""
    from transformers import AutoTokenizer

    # Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with streaming if specified
    print("Loading dataset")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        streaming=args.streaming,
        split="train",
    )

    print("Splitting dataset")
    if args.streaming:
        validation_dataset = dataset.take(args.num_eval_samples)
        temp_dataset = dataset.skip(args.num_eval_samples)
        train_dataset = (
            temp_dataset.take(args.num_train_samples)
            if args.num_train_samples > 0
            else temp_dataset
        )

        from datasets import IterableDatasetDict

        dataset = IterableDatasetDict()
        dataset["train"] = train_dataset
        dataset["test"] = validation_dataset

    else:
        dataset = dataset["train"].train_test_split(
            test_size=args.num_eval_samples,
            train_size=args.num_train_samples,
            shuffle=True,
            seed=args.seed,
        )

    return dataset, tokenizer


metric_accuracy = evaluate.load("accuracy")


# Compute the evaluation metrics
def compute_metrics(eval_pred: EvalPrediction, compute_result=False):
    with torch.no_grad():
        # Get the logits, attention mask, and labels
        logits = eval_pred.predictions.detach()
        metric_labels = eval_pred.label_ids.detach()
        attention_mask = eval_pred.inputs["attention_mask"].detach()

        # Shift the labels and attention mask to the left
        metric_labels = metric_labels[..., 1:]
        attention_mask = attention_mask[..., 1:]
        logits = logits[..., :-1, :]

        predictions = torch.argmax(logits, dim=-1)

        # Mask out the padding tokens
        if attention_mask is not None:
            predictions = predictions * attention_mask
            metric_labels = metric_labels * attention_mask

        # Flatten the input and move to CPU
        metric_labels = metric_labels.flatten().cpu()
        predictions = predictions.flatten().cpu()
        attention_mask = attention_mask.flatten().cpu()

        metric_accuracy.add_batch(predictions=predictions, references=metric_labels)

        del logits, metric_labels, predictions, attention_mask
        torch.cuda.empty_cache()

    if compute_result:
        return {
            "accuracy": metric_accuracy.compute()["accuracy"],
        }
    else:
        return {}


def main():
    print("Processing command-line arguments")
    from transformers import HfArgumentParser

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, SFTConfig, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = (
        parser.parse_args_into_dataclasses()
    )

    training_args.batch_eval_metrics = True
    training_args.include_inputs_for_metrics = True
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
    training_args.dataset_text_field="text" if not training_args.dataset_text_field else training_args.dataset_text_field

    # Prepare dataset
    data_args.max_seq_length = training_args.max_seq_length
    datasets, tokenizer = prepare_dataset(data_args)
    if data_args.streaming and not training_args.max_steps > 0:
        training_args.max_steps = (
            data_args.num_train_samples // training_args.per_device_train_batch_size
        )

    # Initialize model
    print("Initializing model")
    config = get_model_config(model_args)
    # config._attn_implementation = "eager"
    config._attn_implementation = "flash_attention_2"
    # config._attn_implementation = "sdpa"
    config.vocab_size = len(tokenizer)
    model = MistralForCausalLM(config)
    model_num_params = sum(p.numel() for p in model.parameters())
    model_num_params = (
        f"{model_num_params / 1e6:.2f}M"
        if model_num_params > 1e6
        else f"{model_num_params / 1e9:.2f}B"
    )
    print(f"Model has {model_num_params} parameters")

    # Configure wandb logging
    if "wandb" in training_args.report_to:
        print("Configuring wandb logging")
        import os

        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"] = wandb_args.project_name

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = wandb_args.watch

        # log the model
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model

    # Initialize trainer
    print("Initializing trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    try:
        print("Starting training")
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print("Saving the model")
        trainer.save_model()


if __name__ == "__main__":
    main()
