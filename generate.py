import argparse

# Get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from")
parser.add_argument("--model_path", type=str, required=True, help="The path to the model checkpoint")
parser.add_argument("--max_length", type=int, default=50, help="The maximum length of the generated text")
parser.add_argument("--temperature", type=float, default=1.0, help="The temperature for sampling")
parser.add_argument("--device", type=str, default="cpu", help="The device to run the model on")
args = parser.parse_args()

import torch
from transformers import AutoTokenizer, TextStreamer
from transformers.utils import logging

from neon import NeonConfig, NeonForCausalLM

logger = logging.get_logger("neon.modeling_neon")
logger.setLevel(logging.ERROR)

model_path = args.model_path

# Set device
device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise ValueError("CUDA is not available, please run on CPU")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

streamer = TextStreamer(tokenizer)

# Load the model
config = NeonConfig.from_pretrained(model_path)
# Set the model dtype to bfloat16 for better performance
config.torch_dtype = torch.bfloat16
# Set the attention implementation to flash attention 2
config._attn_implementation = "flash_attention_2"
# Load the model from the checkpoint
model = NeonForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
# Switch model to inference mode
model.eval()
# Move model to device
model = model.to(device)

input_text = args.prompt

# If the tokeniser has a chat template, apply it to the input text
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    conversation_history = [{"role": "user", "content": input_text}]
    input_text = tokenizer.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=False)

# Generate text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)
try:
    _ = model.generate(
        input_ids,
        use_cache=False,
        attention_mask=attention_mask,
        streamer=streamer,
        do_sample=True,
        max_length=args.max_length,
        temperature=args.temperature,
    )
except KeyboardInterrupt:
    print("\n\nGeneration interrupted by user")
    exit(0)
