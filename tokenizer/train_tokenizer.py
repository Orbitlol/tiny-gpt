import argparse
import os
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/corpus.txt")
    ap.add_argument("--model_prefix", default="tokenizer/spm/tok")
    ap.add_argument("--vocab_size", type=int, default=300)  # Changed default from 8000 to 300
    ap.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|user|>", "<|assistant|>", "<|end|>"],
        hard_vocab_limit=False,  # Prevents SentencePiece from crashing if data has fewer tokens
    )
    print(f"Tokenizer saved to {args.model_prefix}.model")

if __name__ == "__main__":
    main()
