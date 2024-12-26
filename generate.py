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
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval().to(self.device)

    def generate(self, prompt, max_length=50, temperature=1.0, top_p=0.9, top_k=50, 
                repetition_penalty=1.1, do_sample=True):
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
                do_sample=do_sample,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user")
            return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="Nucleus sampling: probability threshold")
    parser.add_argument("--top_k", type=int, default=50,
                      help="Top-k sampling: k value")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                      help="Penalty for token repetition")
    parser.add_argument("--no_sample", action="store_false", dest="do_sample",
                      help="Disable sampling (use greedy decoding)")
    args = parser.parse_args()

    generator = NeonGenerator(args.model_path, args.device)
    generator.generate(
        args.prompt,
        args.max_length,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.do_sample
    )

if __name__ == "__main__":
    main()
