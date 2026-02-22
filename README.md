#  Alzheimer's AI Support Chatbot

A sequence-to-sequence transformer model designed to support curious individuals, family members, and caregivers seeking information about Alzheimer's disease. The system generates informative, accessible, and compassionate text responses about the condition.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Models](#models)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project builds and evaluates multiple sequence-to-sequence (seq2seq) models for Alzheimer's disease question answering and information generation. The goal is to provide an accessible conversational interface that helps:

- **Family members** understand what their loved ones are experiencing
- **Caregivers** find quick, informative responses to common questions
- **Curious individuals** learn about Alzheimer's disease in plain language

The system was built using both a **custom TensorFlow transformer** trained from scratch and a **pretrained BERT Encoder-Decoder** (bert-base-uncased) fine-tuned on domain-specific Alzheimer's data.

> ⚠️ **Disclaimer**: This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Features

-  **Seq2seq text generation** — generates responses to Alzheimer's-related questions
-  **Hyperparameter tuning** — 6 custom transformer configurations compared
-  **Pretrained BERT** — fine-tuned BERT Encoder-Decoder for stronger language understanding
-  **Comprehensive evaluation** — BLEU, METEOR, F1, and exact match metrics
-  **Full-stack deployment** — Flask REST API backend + HTML/CSS/JS frontend
-  **Chat interface** — clean, accessible UI with conversation history

---

## Project Structure

```
chatbot/
│
├── backend/
│   ├── app.py                        ← Flask API (serves predictions)
│   ├── best_bert_model/              ← Fine-tuned BERT weights
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── special_tokens_map.json
│   ├── data.csv                      ← Training dataset
│   ├── requirements.txt              ← Python dependencies
│   ├── Procfile                      ← Render deployment config
│   └── evaluation_results.json       ← Model evaluation output
│
├── frontend/
│   ├── index.html                    ← Chat interface
│   ├── style.css                     ← Brown & orange theme
│   └── script.js                     ← API calls and chat logic
│
├── notebooks/
│   └── training.ipynb                ← Full training pipeline (Google Colab)
│
└── README.md
```

---

## How It Works

### Architecture

The system uses a **sequence-to-sequence** architecture where:

1. An **encoder** reads and understands the input question
2. A **decoder** generates a response token by token
3. **Attention mechanisms** allow the decoder to focus on relevant parts of the input

### BERT Encoder-Decoder (Production Model)

The production model wraps `bert-base-uncased` as both encoder and decoder using HuggingFace's `EncoderDecoderModel`:

```
Input question
      ↓
BERT Encoder (bidirectional, 12 layers, 768 hidden dim)
      ↓
Contextual representation of input
      ↓
BERT Decoder (with cross-attention to encoder output)
      ↓
Generated response (beam search, 4 beams, no-repeat-ngram=3)
```

BERT's bidirectional pre-training on 3.3 billion words makes it particularly strong at understanding nuanced language around medical conditions, caregiving, and patient experiences.

### Custom Transformer (Training Experiments)

A custom TensorFlow transformer was also built from scratch with the following components:

| Component | Description |
|---|---|
| Positional Embedding | Combines token and position information |
| Multi-Head Self-Attention | Encoder understands full input context |
| Causal Self-Attention | Decoder generates tokens autoregressively |
| Cross-Attention | Decoder attends to encoder output |
| Feed-Forward Layers | Non-linear transformation at each layer |
| Layer Normalization | Stabilizes training at each sub-layer |

---

## Models

### Hyperparameter Experiments — Custom Transformer

Six configurations were trained and compared. Replace the `—` placeholders with your actual results:

| # | Name | Embed Dim | Dense Dim | Heads | Batch | LR | Dropout | Val Loss | Val Acc | Val F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | 256 | 2048 | 8 | 16 | 0.001 | 0.5 | — | — | — |
| 2 | Larger Model | 512 | 4096 | 16 | 8 | 0.001 | 0.3 | — | — | — |
| 3 | Smaller Model | 128 | 1024 | 4 | 16 | 0.001 | 0.5 | — | — | — |
| 4 | Higher LR | 256 | 2048 | 8 | 16 | 0.005 | 0.5 | — | — | — |
| 5 | Lower LR | 256 | 2048 | 8 | 16 | 0.0005 | 0.5 | — | — | — |
| 6 | Higher Dropout | 256 | 2048 | 8 | 16 | 0.001 | 0.7 | — | — | — |


### BERT Fine-tuning Configuration

| Parameter | Value |
|---|---|
| Base model | bert-base-uncased |
| Parameters | 247,014,400 |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Decoding | Beam search (4 beams) |
| No-repeat n-gram | 3 |

---

## Dataset

| Property | Value |
|---|---|
| Domain | Alzheimer's disease information and caregiving Q&A |
| Total samples | ~18,500 |
| Format | Source-target text pairs (CSV) |
| Source vocab size | 25,000 tokens |
| Target vocab size | 25,000 tokens |
| Max sequence length | 200 tokens |

### Preprocessing Pipeline

1. Lowercase all text
2. Remove punctuation (preserving `[` and `]` for special tokens)
3. Add `[start]` and `[end]` tokens to target sequences
4. Tokenize using TensorFlow `TextVectorization`
5. Pad/truncate to fixed sequence length
6. Split into train / validation / test sets

---

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- pip
- Git (with Git LFS for the model file)

### 1. Clone the repository

```bash
git clone https://github.com/edith123321/chatbot.git
cd chatbot
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Verify the model files are present

```bash
ls backend/best_bert_model/
# Should show: config.json, model.safetensors, vocab.txt, etc.
```

---

## Running Locally

### Start the backend server

```bash
cd backend
python app.py
```

Expected output:
```
Loading BERT model...
✓ BERT model loaded on cpu.
 * Running on http://0.0.0.0:5500
 * Running on http://127.0.0.1:5500
```

### Test the health endpoint

```bash
curl http://localhost:5500/health
```

Response:
```json
{
  "status": "ok",
  "model": "BERT Encoder-Decoder",
  "device": "cpu"
}
```

### Test a prediction

```bash
curl -X POST http://localhost:5500/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the early signs of Alzheimers?"}'
```

### Open the frontend

Double-click `frontend/index.html` to open the chat UI in your browser. It connects to `localhost:5500` automatically.


### Handling the Large Model File

`model.safetensors` is ~500MB and exceeds GitHub's 100MB limit. Use one of:

**Option A — Git LFS**
```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git commit -m "Track model with LFS"
git push
```



## API Reference

### `GET /health`

Check if the model is loaded and running.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model": "BERT Encoder-Decoder",
  "device": "cpu"
}
```

---

### `POST /predict`

Generate a response to an Alzheimer's-related question.

**Request:**
```json
{
  "text": "What are the early signs of Alzheimer's disease?"
}
```

**Response `200 OK`:**
```json
{
  "input": "What are the early signs of Alzheimer's disease?",
  "prediction": "Early signs of Alzheimer's include memory loss that disrupts daily life..."
}
```

**Error `400 Bad Request`:**
```json
{
  "error": "Missing \"text\" field in request body"
}
```

**Error `500 Internal Server Error`:**
```json
{
  "error": "Description of what went wrong"
}
```

---

## Frontend

The chat interface was built with accessibility and warmth in mind.

### Design

| Element | Style |
|---|---|
| Bot message bubbles | Deep brown background |
| User message bubbles | Mild orange background |
| Primary font | DM Sans (sans-serif, clean and readable) |
| Display font | Playfair Display (headings) |
| Layout | Sidebar + main chat area |

### Features

- **Conversation history** — multiple sessions saved in the sidebar
- **New conversation** — start fresh at any time
- **Suggestion chips** — quick-start prompts on the empty state
- **Typing indicator** — animated dots while the model generates
- **Auto-resize input** — textarea grows as you type
- **Keyboard shortcuts** — `Enter` to send, `Shift+Enter` for new line
- **Responsive** — works on desktop and mobile



## Ethical Considerations

This project was developed with careful attention to the sensitive nature of Alzheimer's disease as a topic.

### Medical Disclaimer
This model is an **educational tool only**. It does not provide medical advice, diagnosis, or treatment recommendations. All generated responses should be verified with a qualified healthcare professional.

### Hallucination Risk
Like all generative models, this system can occasionally produce confident-sounding but factually incorrect statements. This is an inherent limitation of neural text generation and is why human oversight is essential for any medical application.

### No Data Collection
The deployed application does not store user conversations, collect personally identifiable information, or track usage.

### Emotional Sensitivity
Alzheimer's disease affects patients and families in profound ways. The model was evaluated not only for factual accuracy but for the tone and compassion of its responses.

### Transparency
Users are clearly informed they are interacting with an AI system, not a medical professional or counsellor.

---

## Future Work

- **Domain-specific pretraining** — fine-tune on BioBERT or PubMedBERT for stronger medical language understanding
- **Retrieval-Augmented Generation (RAG)** — ground responses in verified sources such as PubMed and the Alzheimer's Association knowledge base to reduce hallucination
- **Larger dataset** — collect more Alzheimer's-specific Q&A pairs from caregiver forums, clinical guidelines, and patient support resources
- **Response filtering** — post-processing layer to flag potentially harmful or inaccurate outputs before serving to users
- **Accessibility improvements** — larger fonts, high-contrast mode, screen reader support for elderly users
- **Multilingual support** — extend to Spanish, French, and other languages common in caregiving communities
- **User feedback loop** — thumbs up/down rating on responses to gather quality signal for future fine-tuning

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. https://arxiv.org/abs/1810.04805
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL.
- Banerjee, S., & Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments*. ACL Workshop.
- Lee, J., Yoon, W., Kim, S., et al. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. Bioinformatics, 36(4), 1234–1240.
- Gu, Y., Tinn, R., Cheng, H., et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing*. ACM CHIL.
- Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP.
- Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL.
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. https://arxiv.org/abs/1409.0473
- Alzheimer's Association. (2023). *Alzheimer's Disease Facts and Figures*. https://www.alz.org/alzheimers-dementia/facts-figures

---

## License

This project is intended for educational and research purposes only. It is not approved for clinical or diagnostic use.

---

*Built with TensorFlow, PyTorch, HuggingFace Transformers, Flask, and deep care for Alzheimer's patients and their families.*
