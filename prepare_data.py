# prepare_data.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
from datasets import load_dataset
import keras_nlp
import tensorflow as tf


def preprocess_function(examples, tokenizer, max_length=512):
    """Convert question-answer pairs to formatted text and tokenize."""
    
    questions = examples["question"]
    answers = examples["answer"]

    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    prefix_texts = [f"Question: {q}\nAnswer: " for q in questions]
    prefix_tokenized = tokenizer(
        prefix_texts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    prefix_lengths = [len(ids) for ids in prefix_tokenized["input_ids"]]

    sample_weights = []

    for i, length in enumerate(prefix_lengths):
        mask = np.zeros(max_length, dtype=np.float32)

        input_ids = tokenized["input_ids"][i]
        pad_token_id = tokenizer.token_to_id("<pad>") if hasattr(tokenizer, "token_to_id") else 0

        pad_positions = [idx for idx, tok in enumerate(input_ids) if tok == pad_token_id]

        if pad_positions:
            end = pad_positions[0]
        else:
            end = max_length

        mask[length:end] = 1.0
        sample_weights.append(mask.tolist())

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "sample_weight": sample_weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    dataset = dataset.shuffle(seed=args.seed)
    splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    train_dataset = splits["train"]
    val_dataset = splits["test"]

    print("Loading tokenizer...")
    tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")

    print("Tokenizing training set...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=128,  # safer
        remove_columns=train_dataset.column_names,
    )

    print("Tokenizing validation set...")
    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=128,
        remove_columns=val_dataset.column_names,
    )

    os.makedirs("data", exist_ok=True)
    train_tokenized.to_parquet("data/train.parquet")
    val_tokenized.to_parquet("data/val.parquet")

    print(f"Preprocessing complete. Train size: {len(train_tokenized)}, Val size: {len(val_tokenized)}")
    print("Files saved to data/")


if __name__ == "__main__":
    main()
