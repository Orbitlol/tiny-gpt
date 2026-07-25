# Tiny GPT - Complete Colab Notebook

This notebook trains and runs Tiny GPT end-to-end on Google Colab with a T4 GPU.

## Step 1: Setup

```python
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt
!pip install -r requirements.txt
```

## Step 2: Prepare Training Data

```python
from google.colab import files

# Upload your markdown files
print("Upload markdown files (.md files):")
uploaded = files.upload()

# Move to data/raw/
import os
os.makedirs('data/raw', exist_ok=True)
for filename in uploaded:
    os.rename(filename, f'data/raw/{filename}')

print(f"Uploaded {len(uploaded)} files to data/raw/")
```

## Step 3: Data Preparation

```python
# Clean markdown files
!python data_prep/clean_md.py

# Train tokenizer
!python tokenizer/train_tokenizer.py --vocab_size 8000

# Prepare dataset (train/val split)
!python data_prep/prepare_dataset.py
```

## Step 4: Pretraining

```python
!python train/pretrain.py
```

*This will take 2-4 hours on T4. Monitor the loss - should decrease steadily.*

## Step 5: Supervised Fine-Tuning (Optional)

```python
# Edit train/sample_sft_data.jsonl with your own examples first,
# or use the defaults
!python train/sft.py
```

## Step 6: Chat Interface

### Option A: CLI Chat
```python
!python inference/chat.py
```

### Option B: Gradio Web UI (Recommended)
```python
!python inference/gradio_app.py
```

Gradio will automatically generate a public URL.

### Option C: Flask + ngrok
```python
!pip install pyngrok
from pyngrok import ngrok
import subprocess
import threading
import time

def run_flask():
    subprocess.run(['python', 'web/app.py'], capture_output=True)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(2)

public_url = ngrok.connect(5000)
print(f"\nChat at: {public_url}")
```

---

## Quick Start (Pre-trained Model)

If you just want to chat without training:

```python
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt
!pip install -r requirements.txt

# Download pretrained checkpoint (if available)
# Or make sure checkpoints/sft_best.pt exists

# Run Gradio
!python inference/gradio_app.py
```

---

## Tips

- **GPU**: Use T4 (Runtime > Change runtime type > GPU > T4)
- **Training time**: ~2-4 hours for pretraining + ~30 min for SFT
- **Memory**: Should fit in 16GB with default config
- **Data**: Start with 50-100 markdown files for testing
- **Temperature**: Lower = more deterministic, Higher = more creative
- **Top-K**: Controls diversity, 50 is good default

