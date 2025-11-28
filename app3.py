import streamlit as st
from recommend3 import recommend_for_text
import networkx as nx
from pyvis.network import Network
import tempfile
import pandas as pd

st.set_page_config(layout="wide", page_title="Journal Recommender")
st.title("Scopus Journal Recommender")

with st.sidebar:
    st.header("New Manuscript")
    title = st.text_input("Title", "")
    abstract = st.text_area("Abstract", "", height=200)
    use_classifier = st.checkbox("Use XGBoost classifier fusion", value=False)
    paper_level = st.checkbox("Paper-level retrieval (exact matches)", value=False)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = st.slider("Top K results", 5, 30, 15)

if st.sidebar.button("Recommend"):
    if not title.strip():
        st.error("Please provide a title.")
    else:
        text = title.strip() + ". " + (abstract.strip() if abstract.strip().lower() != "not available" else "")
        with st.spinner("Computing recommendations..."):
            results = recommend_for_text(
                text,
                model_name=model_name,
                top_k=top_k,
                use_classifier=use_classifier,
                paper_level=paper_level
            )

        # Table
        rows = []
        for r in results:
            rows.append({
                "paper_title": r["paper_title"],
                "journal": r["journal"],
                "publisher": r["publisher"],
                "scopus_id": r["scopus_id"],
                "score": round(r["score"], 4),
                "confidence": r["confidence"]
            })
        st.subheader("Top Paper Recommendations")
        table = pd.DataFrame(rows)
        st.dataframe(table)

        # # Network
        # # --- Build the graph ---
        # G = nx.Graph()
        # user_node = "USER_PAPER"
        # G.add_node(user_node, size=40, color="red")

        # for r in results:
        #     paper_node = r["paper_title"]
        #     G.add_node(paper_node, size=10, title=paper_node)
        #     G.add_edge(user_node, paper_node, weight=r["score"])

        # # --- Create PyVis Network ---
        # net = Network(height="600px", width="100%", notebook=False)

        # for n, d in G.nodes(data=True):
        #     color = "red" if n == user_node else "lightblue"
        #     net.add_node(n, label=str(n), value=d.get("size", 10), color=color, title=d.get("title", n))

        # for u, v, d in G.edges(data=True):
        #     net.add_edge(u, v, value=d.get("weight", 0.1)*10)

        # # --- Save HTML and render in Streamlit ---
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        #     net.write_html(tmp.name)
        #     html = open(tmp.name, "r", encoding="utf-8").read()
        #     st.components.v1.html(html, height=650, scrolling=True)

        # CSV export
        csv_data = "paper_title,journal,publisher,scopus_id,score,confidence\n"
        for r in results:
            csv_data += f"{r['paper_title']},{r['journal']},{r['publisher']},{r['scopus_id']},{r['score']},{r['confidence']}\n"
        st.download_button("Download recommendations CSV", data=csv_data, file_name="recommendations.csv")
