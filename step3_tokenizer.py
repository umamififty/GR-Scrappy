import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import nltk

# First-time use:
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- Config ---
JSON_DIR = "policy_json"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_SAVE_PATH = "policy_embeddings"

# --- Load Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

# --- Sentence Embedding Helper ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Load and chunk content ---
def load_json_sentences(json_dir):
    data = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    content = obj.get("content", "").strip()
                    if content:
                        sentences = sent_tokenize(content)
                        data.append({
                            "filepath": path,
                            "sentences": sentences
                        })
    return data

# --- Encode sentences ---
def encode_sentences(sentences, batch_size=32):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
        with torch.no_grad():
            model_output = model(**tokens)
        sent_embeds = mean_pooling(model_output, tokens['attention_mask'])
        embeddings.append(sent_embeds.cpu().numpy())
    return np.vstack(embeddings)

# --- Run Full Pipeline ---
def run_pipeline():
    dataset = load_json_sentences(JSON_DIR)
    os.makedirs(EMBED_SAVE_PATH, exist_ok=True)

    for item in tqdm(dataset, desc="Processing policies"):
        sent_list = item["sentences"]
        path = item["filepath"]
        emb = encode_sentences(sent_list)
        fname = os.path.basename(path).replace(".json", ".npy")
        np.save(os.path.join(EMBED_SAVE_PATH, fname), emb)

        # Optional: Save mapping
        with open(os.path.join(EMBED_SAVE_PATH, fname.replace(".npy", "_meta.json")), "w") as f:
            json.dump({
                "filepath": path,
                "num_sentences": len(sent_list)
            }, f)

if __name__ == "__main__":
    run_pipeline()