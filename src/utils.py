# src/utils.py
import json
import pandas as pd
import numpy as np
from typing import List, Dict

def load_json_to_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def safe_list_to_str(x):
    if isinstance(x, list):
        return " ".join([str(i) for i in x if i])
    if pd.isna(x):
        return ""
    return str(x)

def combine_text(row):
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    if not abstract or abstract.lower() == "not available": 
        abstract = ""
    subjects = safe_list_to_str(row.get("subject_area", ""))
    # You can adjust the order/weights by repeating fields if desired
    text = title.strip()
    if abstract:
        text += ". " + abstract.strip()
    if subjects:
        text += ". Subjects: " + subjects
    return text