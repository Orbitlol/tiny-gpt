from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import sentencepiece as spm
from model.config import GPTConfig
from model.gpt import GPT
import os

app = Flask(__name__, template_folder='web', static_folder='web')
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
sp = spm.SentencePieceProcessor(model_file="tokenizer/spm/tok.model")
user_id = sp.piece_to_id("<|user|>")
asst_id = sp.piece_to_id("<|assistant|>")
end_id = sp.piece_to_id("<|end|>")

ckpt_path = "checkpoints/sft_best.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")

ckpt = torch.load(ckpt_path, map_location=device)
mcfg = GPTConfig(**ckpt["config"])
model = GPT(mcfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Model loaded on {device}")
print(f"Model params: {model.num_params():,}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        prompt_ids = [user_id] + sp.encode(user_message, out_type=int) + [asst_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=50, eos_token_id=end_id)

        new_ids = out[0, len(prompt_ids):].tolist()
        if end_id in new_ids:
            new_ids = new_ids[:new_ids.index(end_id)]

        response = sp.decode(new_ids).strip()

        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
