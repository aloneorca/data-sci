# src/embed.py
# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# from tqdm import tqdm

# def encode_and_save(df_pkl="data/df_scopus.pkl", out_npy="data/embeddings_specter.npy", model_name="sentence-transformers/allenai-specter", batch_size=32, device=None):
#     df = pd.read_pickle(df_pkl)
#     model = SentenceTransformer(model_name, device=device)  # device = "cuda" or "cpu"
#     texts = df["text_for_embedding"].fillna("").tolist()
#     # encode
#     embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
#     embeddings = np.asarray(embeddings)
#     np.save(out_npy, embeddings)
#     # attach to df and save
#     df["embedding"] = list(embeddings)
#     df.to_pickle(df_pkl)  # overwrite with embeddings
#     print(f"Saved embeddings to {out_npy} and updated {df_pkl}")

# if __name__ == "__main__":
#     # call with default model. If you have GPU: device="cuda"
#     encode_and_save()

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

