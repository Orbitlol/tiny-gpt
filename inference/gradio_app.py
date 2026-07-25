#!/usr/bin/env python3
"""
Tiny GPT - Gradio Web Interface

Usage:
    python inference/gradio_app.py
    python inference/gradio_app.py --checkpoint checkpoints/sft_best.pt
"""

import gradio as gr
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT
import argparse
import os
import sys

def load_model(checkpoint_path, tokenizer_path):
    """Load model and tokenizer"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    return model, sp, device

def main():
    parser = argparse.ArgumentParser(description="Tiny GPT Gradio Interface")
    parser.add_argument("--checkpoint", default="checkpoints/sft_best.pt")
    parser.add_argument("--spm_model", default="tokenizer/spm/tok.model")
    args = parser.parse_args()

    print("Loading model...")
    try:
        model, sp, device = load_model(args.checkpoint, args.spm_model)
        user_id = sp.piece_to_id("<|user|>")
        asst_id = sp.piece_to_id("<|assistant|>")
        end_id = sp.piece_to_id("<|end|>")
        print(f"Model loaded: {model.num_params():,} parameters on {device}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    def chat(user_message, temperature, top_k, max_tokens):
        """Generate response"""
        if not user_message.strip():
            return "Please enter a message."
        
        try:
            prompt_ids = [user_id] + sp.encode(user_message, out_type=int) + [asst_id]
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                out = model.generate(
                    idx,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=end_id
                )
            
            new_ids = out[0, len(prompt_ids):].tolist()
            if end_id in new_ids:
                new_ids = new_ids[:new_ids.index(end_id)]
            
            response = sp.decode(new_ids).strip()
            return response if response else "(No response generated)"
        except Exception as e:
            return f"Error: {str(e)}"

    # Create interface
    with gr.Blocks(title="Tiny GPT") as demo:
        gr.Markdown("# Tiny GPT Chat")
        gr.Markdown("A lightweight language model trained from scratch.")
        
        with gr.Row():
            message_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=3,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=100)
        
        with gr.Row():
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1,
                scale=1
            )
            top_k = gr.Slider(
                label="Top K",
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                scale=1
            )
            max_tokens = gr.Slider(
                label="Max Tokens",
                minimum=10,
                maximum=500,
                value=200,
                step=10,
                scale=1
            )
        
        response_output = gr.Textbox(
            label="Bot Response",
            interactive=False,
            lines=4
        )
        
        submit_btn.click(
            fn=chat,
            inputs=[message_input, temperature, top_k, max_tokens],
            outputs=response_output
        )
        message_input.submit(
            fn=chat,
            inputs=[message_input, temperature, top_k, max_tokens],
            outputs=response_output
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
