# src/recommend.py
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd

def load_profiles(profiles_path="data/journal_profiles.pkl"):
    """Load journal profiles containing centroid + paper info."""
    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)
    journal_ids = list(profiles.keys())
    centroids = np.stack([profiles[j]["centroid"] for j in journal_ids])
    return profiles, journal_ids, centroids

def embed_text(model, text, device=None):
    if not text or text.lower() == "not available":
        text = ""
    return model.encode([text], device=device)[0]

def recommend_for_text(
    text,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    profiles_path="data/journal_profiles.pkl",
    top_k=15,
    use_classifier=False,
    models_dir="models",
    paper_level=False  # NEW: if True, do paper-level retrieval
):
    model = SentenceTransformer(model_name)
    profiles, journal_ids, centroids = load_profiles(profiles_path)
    emb = embed_text(model, text)

    results = []

    if paper_level:
        # PAPER-LEVEL RETRIEVAL: similarity against all papers in dataset
        all_papers = []
        all_embeddings = []
        for j in journal_ids:
            for p in profiles[j]["papers"]:
                all_papers.append({
                    "title": p.get("title", "N/A"),
                    "abstract": p.get("abstract", "N/A"),
                    "journal": j,
                    "publisher": p.get("publisher", "N/A"),
                    "dc_identifier": p.get("dc_identifier", "N/A"),
                    "embedding": p["embedding"]
                })
                all_embeddings.append(p["embedding"])
        all_embeddings = np.stack(all_embeddings)
        sims = cosine_similarity([emb], all_embeddings).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        for idx in top_idx:
            p = all_papers[idx]
            score = float(sims[idx])
            confidence = "LOW"
            if score > 0.75: confidence = "HIGH"
            elif score > 0.6: confidence = "MEDIUM"

            results.append({
                "paper_title": p["title"],
                "paper_abstract": p["abstract"],
                "journal": p["journal"],
                "publisher": p["publisher"],
                "score": score,
                "confidence": confidence,
                "scopus_id": p["dc_identifier"]
            })

    else:
        # JOURNAL-LEVEL RETRIEVAL (current approach)
        sims = cosine_similarity([emb], centroids)[0]
        idx = sims.argsort()[::-1][:top_k]

        clf_scores = None
        if use_classifier:
            pca = joblib.load(f"{models_dir}/pca.pkl")
            scaler = joblib.load(f"{models_dir}/scaler.pkl")
            clf = joblib.load(f"{models_dir}/xgb_classifier.pkl")
            journal_map = joblib.load(f"{models_dir}/journal_map.pkl")

            emb_pca = pca.transform(emb.reshape(1, -1))
            meta = np.zeros((1, 3))
            X = np.hstack([emb_pca, scaler.transform(meta)])
            probs = clf.predict_proba(X)[0]
            clf_scores = {journal_map[i]: p for i, p in enumerate(probs)}

        for i in idx:
            jid = journal_ids[i]
            sim_score = float(sims[i])
            fused_score = sim_score
            if clf_scores and jid in clf_scores:
                fused_score = 0.5 * sim_score + 0.5 * clf_scores[jid]

            confidence = "LOW"
            if fused_score > 0.75: confidence = "HIGH"
            elif fused_score > 0.6: confidence = "MEDIUM"

            # Top paper per journal
            papers = profiles[jid]["papers"]
            paper_embs = np.stack([p["embedding"] for p in papers])
            paper_sims = cosine_similarity([emb], paper_embs).flatten()
            top_idx = np.argmax(paper_sims)
            top_paper = papers[top_idx]

            results.append({
                "paper_title": top_paper.get("title", "N/A"),
                "paper_abstract": top_paper.get("abstract", "N/A"),
                "journal": jid,
                "publisher": top_paper.get("publisher", "N/A"),
                "score": fused_score,
                "confidence": confidence,
                "scopus_id": top_paper.get("dc_identifier", "N/A")
            })

    return results
