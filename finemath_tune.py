print("Loading imports")
import warnings
from dataclasses import dataclass, field

import evaluate
import torch
from datasets import Dataset, load_dataset
from transformers import EvalPrediction
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from trl import SFTConfig, SFTTrainer

from neon import NeonForCausalLM

warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="HuggingFaceTB/finemath",
        metadata={"help": "Name of the dataset to use"},
    )
    dataset_config_name: str = field(
        default="finemath-4plus",
        metadata={"help": "Name of the dataset configuration to use"},
    )
    num_train_samples: int = field(
        default=1000000, metadata={"help": "Number of training samples"}
    )
    num_eval_samples: int = field(
        default=2048, metadata={"help": "Number of evaluation samples"}
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory instead of memory mapping it"},
    )


@dataclass
class WandbArguments:
    project_name: str = field(
        default=None, metadata={"help": "Name of the W&B project"}
    )
    watch: str = field(
        default="false",
        metadata={
            "help": "Whether to watch the training",
            "choices": ["all", "gradients", "parameters", "false"],
        },
    )
    wandb_log_model: str = field(
        default="false",
        metadata={
            "help": "Whether to log the model",
            "choices": ["end", "checkpoint", "false"],
        },
    )


def prepare_dataset(args: DataArguments):
    """Load and prepare the dataset with support for both regular and streaming modes."""

    print("Loading dataset")
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, keep_in_memory=args.keep_in_memory)
    if args.num_eval_samples > 0:
        if "test" not in dataset:
            if "train" in dataset:
                dataset = dataset["train"].train_test_split(
                    test_size=args.num_eval_samples,
                    shuffle=True,
                    seed=42,
                    keep_in_memory=args.keep_in_memory,
                )
            elif isinstance(dataset, Dataset):
                dataset = dataset.train_test_split(
                    test_size=args.num_eval_samples,
                    shuffle=True,
                    seed=42,
                    keep_in_memory=args.keep_in_memory,
                )
    else:
        ValueError("`--num_eval_samples` must be greater than 0.")

    if args.num_train_samples > 0:
        if len(dataset["train"]) > args.num_train_samples:
            dataset["train"] = dataset["train"].select(range(args.num_train_samples))
        else:
            ValueError("`--num_train_samples` must be less than the number of available training samples.")

    return dataset


def prepare_tokenizer(max_seq_length: int, model_name_or_path: str):
    """Load and prepare the dataset with support for both regular and streaming modes."""
    from transformers import AutoTokenizer

    # Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Set padding and truncation strategies
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.set_truncation_and_padding(
        padding_strategy=PaddingStrategy.LONGEST,
        truncation_strategy=TruncationStrategy.LONGEST_FIRST,
        max_length=max_seq_length,
        stride=max_seq_length // 8,
        pad_to_multiple_of=8,
        padding_side="left",
    )

    return tokenizer


metric_accuracy = evaluate.load("accuracy")


# Compute the evaluation metrics
def compute_metrics(eval_pred: EvalPrediction, compute_result=False):
    with torch.no_grad():
        # Get the logits, attention mask, and labels
        logits = eval_pred.predictions.detach()
        metric_labels = eval_pred.label_ids.detach()
        if hasattr(eval_pred.inputs, "attention_mask"):
            attention_mask = eval_pred.inputs["attention_mask"].detach()
        else:
            attention_mask = None

        # Shift the labels and attention mask to the left
        metric_labels = metric_labels[..., 1:]
        attention_mask = attention_mask[..., 1:] if attention_mask is not None else None
        logits = logits[..., :-1, :]

        predictions = torch.argmax(logits, dim=-1)

        # Mask out the padding tokens
        if attention_mask is not None:
            predictions = predictions * attention_mask
            metric_labels = metric_labels * attention_mask

        # Flatten the input and move to CPU
        metric_labels = metric_labels.flatten().cpu()
        predictions = predictions.flatten().cpu()
        attention_mask = attention_mask.flatten().cpu() if attention_mask is not None else None

        metric_accuracy.add_batch(predictions=predictions, references=metric_labels)

        del logits, metric_labels, predictions, attention_mask

    if compute_result:
        torch.cuda.empty_cache()
        return {
            "accuracy": metric_accuracy.compute()["accuracy"],
        }
    else:
        return {}


def main():
    print("Processing command-line arguments")
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig, WandbArguments))
    model_args, data_args, training_args, wandb_args = (parser.parse_args_into_dataclasses())

    if model_args.model_name_or_path is None:
        raise ValueError("`--model_name_or_path` must be defined.")

    # Configure wandb logging
    if "wandb" in training_args.report_to:
        print("Configuring wandb logging")
        import os

        # set the wandb project where this run will be logged
        if wandb_args.project_name is not None:
            os.environ["WANDB_PROJECT"] = wandb_args.project_name

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = wandb_args.watch

        # log the model
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model

    training_args.run_name = f"{training_args.run_name}-lr{training_args.learning_rate:.2e}-e{training_args.num_train_epochs}-ga{training_args.gradient_accumulation_steps}"
    training_args.batch_eval_metrics = True
    training_args.include_inputs_for_metrics = True
    training_args.include_tokens_per_second = False  # Causes unnecessary dataset reprocessing at start of training, causing a huge delay
    training_args.include_num_input_tokens_seen = True
    training_args.dataset_text_field = (
        "text"
        if not training_args.dataset_text_field
        else training_args.dataset_text_field
    )

    dataset = prepare_dataset(data_args)

    tokenizer = prepare_tokenizer(
        training_args.max_seq_length, model_args.model_name_or_path
    )

    # Initialize model
    print("Initializing model")
    # model_args._attn_implementation = "eager"
    model_args._attn_implementation = "flash_attention_2"
    # model_args._attn_implementation = "sdpa"
    # config = get_model_config(model_args, tokenizer)

    model = NeonForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args._attn_implementation,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    model.config.use_cache = False
    model_num_params = model.num_parameters()
    model_num_params = (
        f"{model_num_params / 1e6:.2f}M"
        if model_num_params > 1e6
        else f"{model_num_params / 1e9:.2f}B"
    )
    print(f"Model has {model_num_params} parameters")

    # Resize model's token embeddings to match the tokenizer
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Initialize trainer
    print("Initializing trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
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
