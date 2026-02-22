import string
import os
import re
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import EncoderDecoderModel, BertTokenizer

app = Flask(__name__)
CORS(app)

# ─── Load BERT model and tokenizer ───────────────────────────────
print("Loading BERT model...")

BERT_MODEL_PATH = 'backend/best_bert_model'  # folder with config.json, model.safetensors etc.

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model     = EncoderDecoderModel.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model.to(device)

print(f"✓ BERT model loaded on {device}.")


# ─── Decode function ─────────────────────────────────────────────
def decode_sequence(input_sentence, max_new_tokens=128):
    enc = bert_tokenizer(
        [input_sentence],
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = bert_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            decoder_start_token_id=bert_tokenizer.cls_token_id,
            eos_token_id=bert_tokenizer.sep_token_id,
            pad_token_id=bert_tokenizer.pad_token_id,
        )

    decoded = bert_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.strip()


# ─── Routes ──────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'BERT Encoder-Decoder', 'device': device})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    input_text = data['text'].strip()
    if not input_text:
        return jsonify({'error': 'Empty input text'}), 400

    try:
        prediction = decode_sequence(input_text)
        return jsonify({'prediction': prediction, 'input': input_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=False)