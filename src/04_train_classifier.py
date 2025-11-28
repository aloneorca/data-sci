# src/train_classifier.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import pickle

def train_xgb(df_pkl="data/df_scopus.pkl", models_dir="models", pca_n=100):
    df = pd.read_pickle(df_pkl)
    df["journal_key"] = df.apply(lambda r: r["issn"] if r["issn"] else r["publication_name"], axis=1)

    counts = df["journal_key"].value_counts()
    valid_journals = counts[counts >= 20].index
    df = df[df["journal_key"].isin(valid_journals)].copy()

    # PCA on embeddings
    X_emb = np.stack(df["embedding"].values)
    pca = PCA(n_components=pca_n, random_state=42)
    X_pca = pca.fit_transform(X_emb)

    # Meta numeric features
    df["author_counted"] = df["author_counted"].fillna(0).astype(int)
    df["reference_count"] = df["reference_count"].fillna(0).astype(int)
    df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
    X_meta = df[["author_counted", "reference_count", "cited_by_count"]].values
    scaler = StandardScaler()
    X_meta = scaler.fit_transform(X_meta)

    X = np.hstack([X_pca, X_meta])
    y = df["journal_key"].astype("category").cat.codes.values
    journal_map = dict(enumerate(df["journal_key"].astype("category").cat.categories))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = xgb.XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="mlogloss", n_estimators=200)
    clf.fit(X_train, y_train)

    joblib.dump(pca, f"{models_dir}/pca.pkl")
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    joblib.dump(clf, f"{models_dir}/xgb_classifier.pkl")
    with open(f"{models_dir}/journal_map.pkl", "wb") as f:
        pickle.dump(journal_map, f)

    print("Saved PCA, scaler, XGB classifier and journal_map")

if __name__ == "__main__":
    train_xgb()
