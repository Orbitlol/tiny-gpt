import argparse
import numpy as np
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/corpus.txt")
    ap.add_argument("--spm_model", default="tokenizer/spm/tok.model")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    text = open(args.input, encoding="utf-8").read()
    ids = np.array(sp.encode(text, out_type=int), dtype=np.uint16)

    split = int(len(ids) * (1 - args.val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    train_ids.tofile(f"{args.out_dir}/train.bin")
    val_ids.tofile(f"{args.out_dir}/val.bin")
    print(f"train tokens: {len(train_ids):,} | val tokens: {len(val_ids):,} | vocab: {sp.vocab_size()}")

if __name__ == "__main__":
    main()
