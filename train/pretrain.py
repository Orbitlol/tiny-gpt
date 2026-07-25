import math, os, yaml
import numpy as np
import torch
from model.config import GPTConfig
from model.gpt import GPT

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_lr(it, warmup, decay_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > decay_iters:
        return min_lr
    ratio = (it - warmup) / (decay_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)

@torch.no_grad()
def evaluate(model, data, mcfg, batch_size, eval_iters, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, mcfg.block_size, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    mcfg, tcfg, scfg = GPTConfig(**cfg["model"]), cfg["pretrain"], cfg["system"]
    device = scfg["device"] if torch.cuda.is_available() else "cpu"
    torch.manual_seed(scfg["seed"])

    train_data = np.memmap("data/processed/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/processed/val.bin", dtype=np.uint16, mode="r")

    model = GPT(mcfg).to(device)
    print(f"model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                   betas=(tcfg["beta1"], tcfg["beta2"]), weight_decay=tcfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(scfg["dtype"] == "float16"))

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for it in range(tcfg["max_iters"]):
        lr = get_lr(it, tcfg["warmup_iters"], tcfg["lr_decay_iters"], tcfg["learning_rate"], tcfg["min_lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(tcfg["grad_accum_steps"]):
            x, y = get_batch(train_data, mcfg.block_size, tcfg["batch_size"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / tcfg["grad_accum_steps"]
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if it % tcfg["eval_interval"] == 0 or it == tcfg["max_iters"] - 1:
            vloss = evaluate(model, val_data, mcfg, tcfg["batch_size"], tcfg["eval_iters"], device)
            print(f"iter {it}: train {loss.item()*tcfg['grad_accum_steps']:.4f} val {vloss:.4f} lr {lr:.2e}")
            if vloss < best_val:
                best_val = vloss
                torch.save({"model": model.state_dict(), "config": mcfg.__dict__}, "checkpoints/pretrain_best.pt")

    print(f"done. best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
