import gradio as gr
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT
import os

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
sp = spm.SentencePieceProcessor(model_file="tokenizer/spm/tok.model")
user_id = sp.piece_to_id("<|user|>")
asst_id = sp.piece_to_id("<|assistant|>")
end_id = sp.piece_to_id("<|end|>")

ckpt_path = "checkpoints/sft_best.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device)
mcfg = GPTConfig(**ckpt["config"])
model = GPT(mcfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Model loaded on {device} | Params: {model.num_params():,}")

def chat(user_message, temperature=0.8, top_k=50):
    """Generate response from user message"""
    if not user_message.strip():
        return "Please enter a message."
    
    try:
        prompt_ids = [user_id] + sp.encode(user_message, out_type=int) + [asst_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=200, temperature=temperature, top_k=top_k, eos_token_id=end_id)
        
        new_ids = out[0, len(prompt_ids):].tolist()
        if end_id in new_ids:
            new_ids = new_ids[:new_ids.index(end_id)]
        
        response = sp.decode(new_ids).strip()
        return response if response else "(No response generated)"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Tiny GPT Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Tiny GPT Chat")
    gr.Markdown("Chat with a lightweight language model trained from scratch.")
    
    with gr.Row():
        with gr.Column(scale=3):
            message_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=2
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1
            )
        with gr.Column(scale=1):
            top_k = gr.Slider(
                label="Top K",
                minimum=1,
                maximum=100,
                value=50,
                step=1
            )
    
    response_output = gr.Textbox(
        label="Response",
        interactive=False,
        lines=4
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[message_input, temperature, top_k],
        outputs=response_output
    )
    message_input.submit(
        fn=chat,
        inputs=[message_input, temperature, top_k],
        outputs=response_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
