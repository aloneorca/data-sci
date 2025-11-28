# src/recommend.py
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib

def load_profiles(profiles_path="data/journal_profiles.pkl"):
    """Load precomputed journal profiles."""
    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)
    journal_ids = list(profiles.keys())
    centroids = np.stack([profiles[j]["centroid"] for j in journal_ids])
    return profiles, journal_ids, centroids

def embed_text(model, text, device=None):
    """Encode text with a SentenceTransformer model."""
    if not text or text.lower() == "not available":
        text = ""
    return model.encode([text], device=device)[0]

def recommend_for_text(
    text, 
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    profiles_path="data/journal_profiles.pkl", 
    top_k=15, 
    use_classifier=False, 
    models_dir="models"
):
    model = SentenceTransformer(model_name)
    profiles, journal_ids, centroids = load_profiles(profiles_path)
    
    emb = embed_text(model, text)
    sims = cosine_similarity([emb], centroids)[0]

    idx = sims.argsort()[::-1][:top_k]
    results = []

    clf_scores = None
    if use_classifier:
        pca = joblib.load(f"{models_dir}/pca.pkl")
        scaler = joblib.load(f"{models_dir}/scaler.pkl")
        clf = joblib.load(f"{models_dir}/xgb_classifier.pkl")
        journal_map = joblib.load(f"{models_dir}/journal_map.pkl")

        emb_pca = pca.transform(emb.reshape(1, -1))
        meta = np.zeros((1, 3))  # author_counted, reference_count, cited_by_count placeholder
        X = np.hstack([emb_pca, scaler.transform(meta)])
        probs = clf.predict_proba(X)[0]

        clf_scores = {journal_map[i]: p for i, p in enumerate(probs)}

    for i in idx:
        jid = journal_ids[i]
        score_text = float(sims[i])
        score = score_text  # no keywords fusion since we dropped author_keywords
        if clf_scores and jid in clf_scores:
            score = 0.5 * score + 0.5 * clf_scores[jid]

        confidence = "LOW"
        if score > 0.75: confidence = "HIGH"
        elif score > 0.6: confidence = "MEDIUM"

        results.append({
            "journal_id": jid,
            "journal_meta": profiles[jid],
            "text_similarity": score_text,
            "fused_score": score,
            "confidence": confidence
        })

    return results

if __name__ == "__main__":
    text = "Example title. Example abstract about convolutional neural networks."
    recs = recommend_for_text(text)
    for r in recs[:10]:
        print(r["journal_id"], r["fused_score"], r["confidence"])
