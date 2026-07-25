# Running Tiny GPT on Google Colab

## Option 1: CLI Chat (No Frontend)

### Setup
```bash
# Clone the repository
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt

# Install dependencies
!pip install -r requirements.txt

# Prepare data (skip if you already have trained model)
# 1. Upload your .md files to data/raw/
# 2. Run data preparation
!python data_prep/clean_md.py
!python tokenizer/train_tokenizer.py --vocab_size 8000
!python data_prep/prepare_dataset.py

# Train (optional, skip if using pretrained checkpoint)
!python train/pretrain.py
!python train/sft.py
```

### Run CLI Chat
```bash
!python inference/chat.py
```

Then interact with the bot in the terminal.

---

## Option 2: Web Frontend with Flask

### Setup
```bash
# Clone the repository
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt

# Install dependencies
!pip install -r requirements.txt flask flask-cors

# Prepare data (same as above if needed)
!python data_prep/clean_md.py
!python tokenizer/train_tokenizer.py --vocab_size 8000
!python data_prep/prepare_dataset.py
!python train/pretrain.py
!python train/sft.py
```

### Run Flask Backend
```bash
# Install ngrok for public URL
!pip install pyngrok

# Run Flask app with ngrok tunnel
from pyngrok import ngrok
import subprocess
import threading

# Start Flask in background
def run_flask():
    subprocess.run(['python', 'web/app.py'])

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Create ngrok tunnel
public_url = ngrok.connect(5000)
print(f"Visit: {public_url}")
```

Then open the public URL in your browser. The chat interface will be available.

---

## Option 3: Web Frontend with Gradio (Simplest)

If you prefer a simpler setup without Flask:

```bash
# Clone and setup
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt
!pip install -r requirements.txt gradio

# Run the Gradio app
!python inference/gradio_app.py
```

This automatically provides a public URL without ngrok.

---

## Quick Start (All-in-One Colab Cell)

```python
# Install and clone
!pip install -r requirements.txt flask flask-cors pyngrok gradio
!git clone https://github.com/Orbitlol/tiny-gpt.git
%cd tiny-gpt

# Assuming you have trained model at checkpoints/sft_best.pt

# Option A: CLI
!python inference/chat.py

# Option B: Flask + ngrok
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
print(f"Chat at: {public_url}")

# Option C: Gradio (recommended)
!python inference/gradio_app.py
```

---

## Uploading Training Data to Colab

1. Create a folder `data/raw/` in the repo
2. In Colab:
   ```python
   from google.colab import files
   files.upload()  # Upload your .md files
   !mv *.md data/raw/
   ```

3. Then run data preparation steps above.

---

## GPU Selection

To use T4 GPU (recommended):
- Go to **Runtime > Change runtime type > GPU > T4**
- The code will auto-detect CUDA

---

## Troubleshooting

**"Checkpoint not found"**
- Make sure `checkpoints/sft_best.pt` exists
- Run training first or upload a pretrained checkpoint

**Out of memory**
- Reduce `batch_size` in `config.yaml`
- Use smaller model: reduce `n_embd` or `n_layer`

**Flask won't load**
- Make sure port 5000 is free
- Check Flask app is running: `curl http://localhost:5000`

