# src/embed.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def encode_and_save(df_pkl="data/df_scopus.pkl", out_npy="data/embeddings.npy",
                    model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, device=None):
    df = pd.read_pickle(df_pkl)
    model = SentenceTransformer(model_name, device=device)
    texts = df["text_for_embedding"].fillna("").tolist()

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings)

    df["embedding"] = list(embeddings)
    df.to_pickle(df_pkl)
    np.save(out_npy, embeddings)
    print(f"Saved embeddings to {out_npy} and updated {df_pkl}")

if __name__ == "__main__":
    encode_and_save()

