# src/preprocess.py
import pandas as pd
import numpy as np
from utils import combine_text
import joblib

def preprocess(in_pkl="data/scopus_data.pkl", out_pkl="data/df_scopus.pkl"):
    df = pd.read_pickle(in_pkl)
    # Normalize columns
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["subject_area"] = df["subject_area"].apply(lambda x: x if isinstance(x, list) else [])
    df["issn"] = df["issn"].fillna("").astype(str)
    df["cited_by_count"] = df["cited_by_count"].replace("", 0).fillna(0).astype(int, errors="ignore")
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").fillna(0).astype(int)
    df["publication_name"] = df["publication_name"].fillna("").astype(str)
    # Flatten classification_code if present
    def extract_codes(c):
        if isinstance(c, list):
            codes = []
            for d in c:
                if isinstance(d, dict):
                    vals = list(d.values())
                    if vals:
                        codes.append(str(vals[0]))
            return codes
        return []
    df["classification_code_flat"] = df.get("classification_code", []).apply(extract_codes) if "classification_code" in df.columns else [[] for _ in range(len(df))]
    # reference counts
    df["reference_count"] = df.get("references", pd.Series([[]]*len(df))).apply(lambda x: len(x) if isinstance(x, list) else 0)
    # text for embedding
    df["text_for_embedding"] = df.apply(combine_text, axis=1)
    df.to_pickle(out_pkl)
    print(f"Saved preprocessed df to {out_pkl}")

if __name__ == "__main__":
    preprocess()
