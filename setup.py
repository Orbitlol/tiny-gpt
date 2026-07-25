#!/usr/bin/env python3
"""
Quick setup script for Tiny GPT
Runs all data preparation and training steps automatically
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        sys.exit(1)
    print(f"✓ {description} completed")

def main():
    print("\nTiny GPT - Automatic Setup")
    print("This will prepare data, train tokenizer, and train the model.\n")
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("tokenizer/spm", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Step 1: Clean markdown
    run_command(
        "python data_prep/clean_md.py",
        "Cleaning markdown files"
    )
    
    # Step 2: Train tokenizer
    run_command(
        "python tokenizer/train_tokenizer.py --vocab_size 2000",
        "Training SentencePiece tokenizer"
    )
    
    # Step 3: Prepare dataset
    run_command(
        "python data_prep/prepare_dataset.py",
        "Preparing train/val datasets"
    )
    
    # Step 4: Pretrain
    run_command(
        "python train/pretrain.py",
        "Pretraining model (this may take 30-60 minutes)"
    )
    
    # Step 5: SFT
    run_command(
        "python train/sft.py",
        "Supervised fine-tuning (this may take 10-20 minutes)"
    )
    
    print(f"\n{'='*60}")
    print("✓ Setup complete! Model is ready.")
    print(f"{'='*60}")
    print("\nNext, run: python inference/gradio_app.py")
    print("Or: python inference/chat.py\n")

if __name__ == "__main__":
    main()
