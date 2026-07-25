# Tiny GPT

A complete, minimal from-scratch implementation of a small GPT language model (10–30M parameters) designed to run on a single NVIDIA Tesla T4 GPU.

**Target**: ~13.8M param GPT, trained on a small plain-English `.md` corpus, pretraining + lightweight chat SFT + optional JSON output, tuned for a single **NVIDIA Tesla T4 (16GB, Turing, no native bf16 tensor cores)**, no distributed training, minimal dependencies, weekend timeline.

## Features

- **Minimal dependencies**: torch, sentencepiece, numpy, tqdm, PyYAML only
- **T4-optimized**: FP16 mixed precision with `GradScaler`, efficient attention backend
- **Complete pipeline**: Data cleaning → tokenizer training → pretraining → SFT → inference
- **Runnable in ~48 hours** on a single T4 GPU

## Quick Start

```bash
pip install -r requirements.txt

# Day 1: Data + Tokenizer + Pretrain
cp your_docs/*.md data/raw/
python data_prep/clean_md.py
python tokenizer/train_tokenizer.py --vocab_size 8000
python data_prep/prepare_dataset.py
python train/pretrain.py

# Day 2: SFT + Inference
# (edit train/sample_sft_data.jsonl with your examples first)
python train/sft.py
python inference/chat.py
```

## Architecture

- **6 transformer blocks** (configurable to 8 for ~30M params)
- Causal self-attention with `F.scaled_dot_product_attention`
- Weight-tied embeddings
- Configurable via `config.yaml`

## Folder Structure

```
tiny-gpt/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                  # Your .md files
│   └── processed/            # corpus.txt, train.bin, val.bin
├── tokenizer/
│   ├── train_tokenizer.py
│   └── spm/                  # tok.model, tok.vocab (generated)
├── model/
│   ├── config.py
│   └── gpt.py
├── data_prep/
│   ├── clean_md.py
│   └── prepare_dataset.py
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   └── sample_sft_data.jsonl
├── inference/
│   └── chat.py
└── checkpoints/              # saved .pt files
```

## Configuration

Edit `config.yaml` to adjust:
- **Model size**: `n_embd`, `n_layer`, `vocab_size`
- **Training**: batch size, learning rates, warmup, decay
- **Hardware**: `dtype` (float16 for T4), `compile` (False by default)

Default config trains a **~13.8M param model**. Scale to 30M by bumping `n_embd` to 512 and/or `n_layer` to 8 (if corpus is large enough).

## Hardware Notes

- **16GB VRAM (Turing/T4)**: Use FP16 mixed precision with `GradScaler` (bf16 arrived with Ampere)
- **Scaled dot product attention**: Auto-selects backend (no flash-attention library needed)
- **torch.compile**: Off by default (can be flaky on older T4 drivers)

## Overfitting Mitigation

- Dropout 0.1, weight decay 0.1, gradient clipping 1.0
- Checkpoints save on **best val loss** (early stopping)
- Watch train/val divergence; scale model only if corpus has enough tokens (rule: tokens ≳ param count)

## JSON Output

The SFT examples teach the model to emit JSON when instructed. At 10–30M params, this is best-effort (no grammar constraints). For reliability, add validation retry in `chat.py`: try `json.loads()` and re-generate if it fails.

## References

- Based on minimal, T4-optimized GPT training from scratch
- Full runnable spec included in codebase
- Suitable for weekend projects or educational purposes
