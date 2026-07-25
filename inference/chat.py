#!/usr/bin/env python3
"""
Tiny GPT - CLI Chat Interface

Usage:
    python inference/chat.py
    python inference/chat.py --checkpoint checkpoints/sft_best.pt --temperature 0.7
"""

import argparse
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Chat with Tiny GPT")
    parser.add_argument("--checkpoint", default="checkpoints/sft_best.pt", help="Path to model checkpoint")
    parser.add_argument("--spm_model", default="tokenizer/spm/tok.model", help="Path to SentencePiece model")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or provide a valid checkpoint path.")
        sys.exit(1)
    
    if not os.path.exists(args.spm_model):
        print(f"Error: Tokenizer model not found at {args.spm_model}")
        print("Please run tokenizer training first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    user_id = sp.piece_to_id("<|user|>")
    asst_id = sp.piece_to_id("<|assistant|>")
    end_id = sp.piece_to_id("<|end|>")

    ckpt = torch.load(args.checkpoint, map_location=device)
    mcfg = GPTConfig(**ckpt["config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Model loaded: {model.num_params():,} parameters")
    print(f"Temperature: {args.temperature}, Top-K: {args.top_k}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_in = input("You> ").strip()
            if not user_in:
                continue
            if user_in.lower() in ["quit", "exit"]:
                break

            prompt_ids = [user_id] + sp.encode(user_in, out_type=int) + [asst_id]
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                out = model.generate(
                    idx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    eos_token_id=end_id
                )
            
            new_ids = out[0, len(prompt_ids):].tolist()
            if end_id in new_ids:
                new_ids = new_ids[:new_ids.index(end_id)]
            
            response = sp.decode(new_ids).strip()
            print(f"Bot> {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
