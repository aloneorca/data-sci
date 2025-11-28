# app.py
import streamlit as st
from recommend2 import recommend_for_text
import networkx as nx
from pyvis.network import Network
import tempfile
import pickle

st.set_page_config(layout="wide", page_title="Journal Recommender")
st.title("Scopus Journal Recommender")

# Sidebar inputs
with st.sidebar:
    st.header("New Manuscript")
    title = st.text_input("Title", "")
    abstract = st.text_area("Abstract", "", height=200)
    use_classifier = st.checkbox("Use XGBoost classifier fusion", value=False)
    model_name = st.selectbox("Embedding Model", ["sentence-transformers/allenai-specter", "sentence-transformers/all-MiniLM-L6-v2"])
    top_k = st.slider("Top K journals", 5, 30, 15)

if st.button("Recommend"):
    if not title.strip():
        st.error("Please provide a title (and optionally an abstract).")
    else:
        text = title.strip() + ". " + (abstract.strip() if abstract.strip().lower() != "not available" else "")
        with st.spinner("Computing recommendations..."):
            results = recommend_for_text(text, model_name=model_name, top_k=top_k, use_classifier=use_classifier)

        # Display table
        rows = []
        for r in results:
            meta = r["journal_meta"]
            rows.append({
                "journal": meta.get("venue"),
                "paper_count": meta.get("paper_count"),
                "avg_citations": round(meta.get("avg_citations", 0), 2),
                "score": round(r["fused_score"], 4),
                "confidence": r["confidence"],
                "top_subjects": ", ".join(meta.get("top_subjects", [])[:5])
            })
        st.subheader("Top Recommendations")
        st.table(rows)

        # Confidence Network Map
        st.subheader("Confidence Network Map")
        G = nx.Graph()
        user_node = "USER_PAPER"
        G.add_node(user_node, size=40, color="red")

        # add recommended journals
        for r in results:
            jid = r["journal_id"]
            meta = r["journal_meta"]
            G.add_node(jid, size=max(10, meta.get("paper_count", 1)), title=jid)
            G.add_edge(user_node, jid, weight=r["fused_score"])

        # Create PyVis network
        net = Network(height="600px", width="100%", notebook=False)
        for n, d in G.nodes(data=True):
            net.add_node(n, label=str(n), title=str(d.get("title", n)), value=d.get("size", 10),
                         color="red" if n == user_node else "lightblue")
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, value=d.get("weight", 0.1) * 10)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.show(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
        st.components.v1.html(html, height=650, scrolling=True)

        # Download CSV
        csv_data = "journal,paper_count,avg_citations,score,confidence\n"
        for r in results:
            meta = r["journal_meta"]
            csv_data += f"{meta.get('venue')},{meta.get('paper_count')},{meta.get('avg_citations')},{r['fused_score']},{r['confidence']}\n"
        st.download_button("Download recommendations CSV", data=csv_data, file_name="recommendations.csv")
