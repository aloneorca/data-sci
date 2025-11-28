# src/build_profiles.py
import numpy as np
import pickle
from collections import Counter
import pandas as pd

def build_journal_profiles(df_pkl="data/df_scopus.pkl", out_profiles="data/journal_profiles.pkl"):
    df = pd.read_pickle(df_pkl)

    # Journal key: ISSN fallback to publication_name
    def venue_key(row):
        if row.get("issn"):
            return row["issn"]
        elif row.get("publication_name"):
            return row["publication_name"]
        else:
            return "UNKNOWN_VENUE"

    df["journal_key"] = df.apply(venue_key, axis=1)
    grouped = df.groupby("journal_key")

    profiles = {}
    for journal, group in grouped:
        embeddings = list(group["embedding"].values)
        if not embeddings:
            continue
        centroid = np.mean(np.stack(embeddings), axis=0)

        # Top subjects
        all_subj = [s for subs in group["subject_area"] for s in (subs if isinstance(subs, list) else [])]
        top_subjects = [s for s,_ in Counter(all_subj).most_common(10)]

        profiles[journal] = {
            "centroid": centroid,
            "paper_count": len(group),
            "avg_citations": float(group["cited_by_count"].mean()),
            "top_subjects": top_subjects,
            "venue": journal
        }

    with open(out_profiles, "wb") as f:
        pickle.dump(profiles, f)
    print(f"Saved {len(profiles)} journal profiles to {out_profiles}")

if __name__ == "__main__":
    build_journal_profiles()
