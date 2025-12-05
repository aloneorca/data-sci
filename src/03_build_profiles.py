# src/build_profiles.py
import numpy as np
import pickle
import pandas as pd
from collections import Counter

def build_journal_profiles(df_pkl="data/df_scopus.pkl", out_profiles="data/journal_profiles.pkl"):
    df = pd.read_pickle(df_pkl)

    # Define journal key
    df["journal_key"] = df.apply(lambda r: r["issn"] if r["issn"] else r["publication_name"], axis=1)
    grouped = df.groupby("journal_key")

    profiles = {}
    for journal, group in grouped:
        embeddings = list(group["embedding"].values)
        if not embeddings:
            continue
        centroid = np.mean(np.stack(embeddings), axis=0)

        # Top subjects (optional)
        all_subj = [s for subs in group["subject_area"] for s in (subs if isinstance(subs, list) else [])]
        top_subjects = [s for s,_ in Counter(all_subj).most_common(10)]

        profiles[journal] = {
            "centroid": centroid,
            "paper_count": len(group),
            "avg_citations": float(group["cited_by_count"].mean()),
            "top_subjects": top_subjects,
            "papers": group[["title", "abstract", "cited_by_count", "dc_identifier", "publisher", "embedding"]].to_dict(orient="records")
        }

    with open(out_profiles, "wb") as f:
        pickle.dump(profiles, f)
    print(f"Saved {len(profiles)} journal profiles to {out_profiles}")


if __name__ == "__main__":
    build_journal_profiles()
