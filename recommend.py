import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def recommend_for_text(
    text,
    df_pkl="data/df_scopus.pkl",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=15
):
    df = pd.read_pickle(df_pkl)
    model = SentenceTransformer(model_name)

    # handle "Not available"
    if not text or text.lower() == "not available":
        text = ""

    emb = model.encode([text])[0]
    sims = cosine_similarity([emb], np.stack(df["embedding"].values))[0]

    idx = sims.argsort()[::-1][:top_k]
    results = []

    for i in idx:
        row = df.iloc[i]
        score = float(sims[i])
        confidence = "LOW"
        if score > 0.75:
            confidence = "HIGH"
        elif score > 0.6:
            confidence = "MEDIUM"

        results.append({
            "paper_title": row.get("title"),
            "journal": row.get("issn") or row.get("publication_name"),
            "score": score,
            "confidence": confidence,
            "embedding": row.get("embedding")
        })
    return results
