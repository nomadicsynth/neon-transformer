import argparse
import torch
from transformers import AutoTokenizer, TextStreamer
from transformers.utils import logging
from neon import NeonConfig, NeonForCausalLM

class NeonGenerator:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, please run on CPU")
        
        # Suppress verbose logging
        logger = logging.get_logger("neon.modeling_neon")
        logger.setLevel(logging.ERROR)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.streamer = TextStreamer(self.tokenizer)
        
        config = NeonConfig.from_pretrained(model_path)
        config.torch_dtype = torch.bfloat16
        config._attn_implementation = "flash_attention_2"
        
        self.model = NeonForCausalLM.from_pretrained(
            model_path, 
            config=config, 
            torch_dtype=torch.bfloat16
        ).eval().to(self.device)

    def generate(self, prompt, max_length=50, temperature=1.0):
        # Apply chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            conversation_history = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                conversation_history, 
                add_generation_prompt=True, 
                tokenize=False
            )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        try:
            _ = self.model.generate(
                input_ids,
                use_cache=False,
                attention_mask=attention_mask,
                streamer=self.streamer,
                do_sample=True,
                max_length=max_length,
                temperature=temperature,
            )
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user")
            return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    generator = NeonGenerator(args.model_path, args.device)
    generator.generate(args.prompt, args.max_length, args.temperature)

if __name__ == "__main__":
    main()
