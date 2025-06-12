# import os
# import json
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# import torch
# import re

# # --- Config ---
# JSON_DIR = "policy_json"
# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMBED_SAVE_PATH = "policy_embeddings"

# # --- Load Model ---
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

# # --- Mean Pooling Helper ---
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
#     return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# # --- Section Splitter ---
# def split_into_sections_with_titles(pages):
#     full_text = "\n".join(pages)
#     pattern = r"\n(?=(?P<header>(?:Chapter|CHAPTER|Annex|ANNEX|Section|SECTION)\s+[^\n]+))"
#     parts = re.split(pattern, full_text)

#     sections = []
#     i = 1 if not re.match(r"^(Chapter|CHAPTER|Annex|ANNEX|Section|SECTION)", parts[0].strip()) else 0
#     while i < len(parts) - 1:
#         title = parts[i].strip()
#         body = parts[i + 1].strip()
#         if body:
#             sections.append((title, body))
#         i += 2
#     return sections

# # --- Load JSONs with Section Splitting ---
# def load_json_sections(json_dir):
#     data = []
#     for root, _, files in os.walk(json_dir):
#         for file in files:
#             if not file.endswith(".json"):
#                 continue
#             path = os.path.join(root, file)
#             with open(path, "r", encoding="utf-8") as f:
#                 obj = json.load(f)
#                 pages = obj.get("pages", [])
#                 if not pages:
#                     continue
#                 sections = split_into_sections_with_titles(pages)
#                 for idx, (title, content) in enumerate(sections):
#                     data.append({
#                         "filepath": path,
#                         "section_index": idx,
#                         "section_title": title,
#                         "section_text": content
#                     })
#     return data

# # --- Embed Sections ---
# def encode_sections(data_items, batch_size=8):
#     os.makedirs(EMBED_SAVE_PATH, exist_ok=True)
#     for item in tqdm(data_items, desc="Embedding sections"):
#         section_text = item["section_text"]
#         title = item["section_title"]
#         tokens = tokenizer(section_text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
#         with torch.no_grad():
#             model_output = model(**tokens)
#         embedding = mean_pooling(model_output, tokens['attention_mask']).cpu().numpy()

#         # Save embedding
#         base = os.path.basename(item["filepath"]).replace(".json", "")
#         section_file = f"{base}_section{item['section_index']:02d}.npy"
#         np.save(os.path.join(EMBED_SAVE_PATH, section_file), embedding)

#         # Save metadata
#         meta = {
#             "original_file": item["filepath"],
#             "section_index": item["section_index"],
#             "section_title": title,
#             "token_length": len(tokens["input_ids"][0])
#         }
#         with open(os.path.join(EMBED_SAVE_PATH, section_file.replace(".npy", "_meta.json")), "w") as f:
#             json.dump(meta, f, indent=2)

# # --- Run ---
# def run_pipeline():
#     data = load_json_sections(JSON_DIR)
#     encode_sections(data)

# if __name__ == "__main__":
#     run_pipeline()

import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

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

# --- Load and chunk content by section/paragraph ---
def load_and_chunk_content(json_dir, min_chunk_length=40):
    data = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    content = obj.get("content", "").strip()
                    if content:
                        # Split by double newline to get paragraphs/sections
                        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                        
                        # Filter out chunks that are too short
                        long_chunks = [chunk for chunk in chunks if len(chunk) > min_chunk_length]
                        
                        data.append({
                            "filepath": path,
                            "chunks": long_chunks
                        })
    return data

# --- Encode chunks ---
def encode_chunks(chunks, batch_size=16):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
        with torch.no_grad():
            model_output = model(**tokens)
        chunk_embeds = mean_pooling(model_output, tokens['attention_mask'])
        embeddings.append(chunk_embeds.cpu().numpy())
    return np.vstack(embeddings)

# --- Run Full Pipeline ---
def run_pipeline():
    dataset = load_and_chunk_content(JSON_DIR)
    os.makedirs(EMBED_SAVE_PATH, exist_ok=True)

    for item in tqdm(dataset, desc="Processing policies"):
        chunk_list = item["chunks"]
        if not chunk_list:
            continue
            
        path = item["filepath"]
        emb = encode_chunks(chunk_list)
        fname = os.path.basename(path).replace(".json", ".npy")
        np.save(os.path.join(EMBED_SAVE_PATH, fname), emb)

        # Optional: Save mapping
        with open(os.path.join(EMBED_SAVE_PATH, fname.replace(".npy", "_meta.json")), "w") as f:
            json.dump({
                "filepath": path,
                "num_chunks": len(chunk_list)
            }, f)

if __name__ == "__main__":
    run_pipeline()