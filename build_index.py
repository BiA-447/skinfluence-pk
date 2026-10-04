"""
build_index.py  -  RUN THIS ONCE (in Google Colab), not inside the Streamlit app.

It reads the product CSV and the knowledge/*.md files, cuts them into chunks,
turns every chunk into an embedding, and saves everything permanently:
    index/faiss.index   -> the FAISS vector index
    index/chunks.json   -> the text + details of every chunk (same order as the index)

Run it again ONLY when you change the CSV or the knowledge files.
"""
import glob
import json
import os
import re

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # free, small, runs on CPU
CSV_PATH = "data/products.csv"
KNOWLEDGE_GLOB = "knowledge/*.md"
INDEX_DIR = "index"


def parse_list(value):
    """Turns  {"Dry","Oily"}  (or a plain 'Dry, Oily') into ['Dry', 'Oily']."""
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    quoted = re.findall(r'"([^"]+)"', s)
    if quoted:
        return [q.strip() for q in quoted]
    return [p.strip() for p in s.strip("{}").split(",") if p.strip()]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_chunks():
    chunks = []

    # ---------- 1) PRODUCTS: one chunk per product ----------
    df = pd.read_csv(CSV_PATH).fillna("")
    df = df[df["is_active"].astype(str).str.lower() == "true"].reset_index(drop=True)

    for i, r in df.iterrows():
        meta = {
            "product_id": f"P{i + 1:03d}",
            "name": str(r["name"]).strip(),
            "brand": str(r["brand"]).strip(),
            "category": str(r["category"]).strip(),
            "product_type": str(r["product_type"]).strip(),
            "origin": str(r["origin"]).strip(),            # "Pakistan" or "Imported"
            "price": to_float(r["price"]),
            "currency": str(r["currency"]).strip() or "PKR",
            "description": str(r["description"]).strip(),
            "skin_types": parse_list(r["skin_types"]),
            "concerns": parse_list(r["concerns"]),
            "goals": parse_list(r["goals"]),
            "key_ingredients": parse_list(r["key_ingredients"]),
            "sensitivity_level": str(r["sensitivity_level"]).strip(),
            "last_verified": str(r["last_verified"]).strip(),
            # optional columns - only filled if your CSV has them
            "seller": str(r["seller"]).strip() if "seller" in df.columns else "",
            "purchase_url": str(r["purchase_url"]).strip() if "purchase_url" in df.columns else "",
        }
        text = (
            f"{meta['name']} by {meta['brand']}. "
            f"Category: {meta['category']} ({meta['product_type']}). "
            f"Origin: {meta['origin']}. "
            f"Key ingredients: {', '.join(meta['key_ingredients'])}. "
            f"Suitable skin types: {', '.join(meta['skin_types'])}. "
            f"Helps with: {', '.join(meta['concerns'])}. "
            f"Supports goals: {', '.join(meta['goals'])}. "
            f"Sensitivity level: {meta['sensitivity_level']}. "
            f"{meta['description']}"
        )
        chunks.append({"type": "product", "text": text, "meta": meta})

    # ---------- 2) KNOWLEDGE: one chunk per '## heading' section ----------
    for path in sorted(glob.glob(KNOWLEDGE_GLOB)):
        raw = open(path, encoding="utf-8").read()
        for section in re.split(r"\n(?=## )", raw):
            section = section.strip()
            if section.startswith("## ") and len(section) > 40:
                chunks.append({"type": "knowledge", "text": section,
                               "meta": {"source": os.path.basename(path)}})
    return chunks


def main():
    chunks = build_chunks()
    n_prod = sum(c["type"] == "product" for c in chunks)
    print(f"{n_prod} product chunks + {len(chunks) - n_prod} knowledge chunks")

    # ---------- 3) EMBED EVERYTHING (this is the only time it happens) ----------
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")

    # ---------- 4) FAISS: inner product on normalised vectors = cosine similarity ----------
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # ---------- 5) SAVE PERMANENTLY ----------
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "chunks": chunks}, f, ensure_ascii=False)
    print("Saved index/faiss.index and index/chunks.json")

    # ---------- 6) QUICK SELF-TEST ----------
    q = model.encode(["oily acne-prone skin needs a lightweight sunscreen"],
                     normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, 3)
    print("\nSelf-test (top 3 for 'oily acne-prone skin needs a lightweight sunscreen'):")
    for s, i in zip(scores[0], ids[0]):
        print(f"  {s:.3f}  {chunks[i]['text'][:90]}...")


if __name__ == "__main__":
    main()
