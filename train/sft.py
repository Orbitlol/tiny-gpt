import json, yaml
import numpy as np
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT

def load_sft_examples(path, sp, block_size):
    user_id, asst_id, end_id = sp.piece_to_id("<|user|>"), sp.piece_to_id("<|assistant|>"), sp.piece_to_id("<|end|>")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt_ids = [user_id] + sp.encode(ex["instruction"], out_type=int) + [asst_id]
            response_ids = sp.encode(ex["response"], out_type=int) + [end_id]
            ids = (prompt_ids + response_ids)[:block_size]
            labels = ([-1] * len(prompt_ids) + response_ids)[:block_size]
            if len(ids) < block_size:
                pad = block_size - len(ids)
                ids += [0] * pad
                labels += [-1] * pad
            examples.append((ids, labels))
    return examples

def get_batch(examples, batch_size, device):
    idxs = np.random.randint(0, len(examples), size=batch_size)
    x = torch.tensor([examples[i][0] for i in idxs], dtype=torch.long)
    y = torch.tensor([examples[i][1] for i in idxs], dtype=torch.long)
    return x.to(device), y.to(device)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    mcfg, tcfg, scfg = GPTConfig(**cfg["model"]), cfg["sft"], cfg["system"]
    device = scfg["device"] if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor(model_file="tokenizer/spm/tok.model")
    examples = load_sft_examples("train/sample_sft_data.jsonl", sp, mcfg.block_size)
    print(f"loaded {len(examples)} SFT examples")

    model = GPT(mcfg).to(device)
    ckpt = torch.load("checkpoints/pretrain_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(scfg["dtype"] == "float16"))

    best_loss = float("inf")
    for it in range(tcfg["max_iters"]):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(tcfg["grad_accum_steps"]):
            x, y = get_batch(examples, tcfg["batch_size"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / tcfg["grad_accum_steps"]
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if it % tcfg["eval_interval"] == 0:
            cur = loss.item() * tcfg["grad_accum_steps"]
            print(f"sft iter {it}: loss {cur:.4f}")
            if cur < best_loss:
                best_loss = cur
                torch.save({"model": model.state_dict(), "config": mcfg.__dict__}, "checkpoints/sft_best.pt")

    print(f"done. best sft loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
