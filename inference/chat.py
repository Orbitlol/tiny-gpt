import argparse
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sft_best.pt")
    ap.add_argument("--spm_model", default="tokenizer/spm/tok.model")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    user_id, asst_id, end_id = sp.piece_to_id("<|user|>"), sp.piece_to_id("<|assistant|>"), sp.piece_to_id("<|end|>")

    ckpt = torch.load(args.checkpoint, map_location=device)
    mcfg = GPTConfig(**ckpt["config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Tiny GPT chat. Ctrl+C to exit.")
    while True:
        try:
            user_in = input("\nyou> ")
        except (KeyboardInterrupt, EOFError):
            break
        prompt_ids = [user_id] + sp.encode(user_in, out_type=int) + [asst_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_token_id=end_id)
        new_ids = out[0, len(prompt_ids):].tolist()
        if end_id in new_ids:
            new_ids = new_ids[:new_ids.index(end_id)]
        print("bot>", sp.decode(new_ids))

if __name__ == "__main__":
    main()
