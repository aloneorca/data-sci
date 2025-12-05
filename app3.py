# import streamlit as st
# from recommend3 import recommend_for_text
# import networkx as nx
# from pyvis.network import Network
# import tempfile
# import pandas as pd

# st.set_page_config(layout="wide", page_title="Journal Recommender")
# st.title("Scopus Journal Recommender")

# with st.sidebar:
#     st.header("New Manuscript")
#     title = st.text_input("Title", "")
#     abstract = st.text_area("Abstract", "", height=200)
#     use_classifier = st.checkbox("Use XGBoost classifier fusion", value=False)
#     paper_level = st.checkbox("Paper-level retrieval (exact matches)", value=False)
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     top_k = st.slider("Top K results", 5, 30, 15)

# if st.sidebar.button("Recommend"):
#     if not title.strip():
#         st.error("Please provide a title.")
#     else:
#         text = title.strip() + ". " + (abstract.strip() if abstract.strip().lower() != "not available" else "")
#         with st.spinner("Computing recommendations..."):
#             results = recommend_for_text(
#                 text,
#                 model_name=model_name,
#                 top_k=top_k,
#                 use_classifier=use_classifier,
#                 paper_level=paper_level
#             )

#         # Table
#         rows = []
#         for r in results:
#             rows.append({
#                 "paper_title": r["paper_title"],
#                 "journal": r["journal"],
#                 "publisher": r["publisher"],
#                 "scopus_id": r["scopus_id"],
#                 "score": round(r["score"], 4),
#                 "confidence": r["confidence"]
#             })
#         st.subheader("Top Paper Recommendations")
#         table = pd.DataFrame(rows)
#         st.dataframe(table)

        
#         st.subheader("Recommendation Summary")

#         # Simple metrics
#         col1, col2 = st.columns(2)
#         col1.metric("Top Journal", table.iloc[0]["journal"])
#         col2.metric("Highest Score", f"{table['score'].max():.3f}")

#         # Tabs for structured layout
#         tab1, tab2, tab3 = st.tabs(["Table", "Similarity Network", "Statistics"])

#         # --- TAB 1: Table ---
#         with tab1:
#             st.subheader("Top Recommendations")
#             st.dataframe(table, use_container_width=True)

#         # --- TAB 2: Network ---
#         with tab2:
#             st.subheader("Journal Similarity Network")

#             # Build the network graph
#             net = Network(height="700px", width="100%", directed=False)
#             net.force_atlas_2based()

#             added = set()
#             sorted_rows = table.sort_values("score", ascending=False)

#             for _, row in sorted_rows.iterrows():
#                 journal = row["journal"]
#                 paper = row["paper_title"]
#                 score = row["score"]

#                 if journal not in added:
#                     net.add_node(journal, label=journal)
#                     added.add(journal)

#                 if paper not in added:
#                     net.add_node(paper, label=paper)
#                     added.add(paper)

#                 net.add_edge(paper, journal, value=score, title=str(score))

#             # Render inside Streamlit
#             import tempfile

#             with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#                 net.write_html(tmp.name)
#                 graph_html = open(tmp.name, "r").read()

#             st.components.v1.html(graph_html, height=700, scrolling=True)

#         # --- TAB 3: Stats ---
#         with tab3:
#             st.subheader("Confidence Values")
#             st.line_chart(table["score"])


#         st.subheader("Journal Similarity Network (by Confidence)")

#         # Create a Network graph
#         net = Network(height="700px", width="100%", directed=False)
#         net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=200)

#         # Add nodes + edges
#         added = set()

#         # Sort by confidence descending
#         sorted_rows = table.sort_values("score", ascending=False)

#         for _, row in sorted_rows.iterrows():
#             journal = row["journal"]
#             paper = row["paper_title"]
#             score = row["score"]

#             # Add nodes (once)
#             if journal not in added:
#                 net.add_node(journal, label=journal, title=journal, shape="dot")
#                 added.add(journal)

#             if paper not in added:
#                 net.add_node(paper, label=paper, title=paper, shape="ellipse")
#                 added.add(paper)

#             # Add edge between paper and journal
#             net.add_edge(
#                 paper,
#                 journal,
#                 value=score,
#                 title=f"score={score}",
#                 color="rgba(50,50,200,0.6)"
#             )

#         # Save & display interactive graph
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#             net.write_html(tmp.name)

#             html_file = tmp.name

#         st.components.v1.html(open(html_file, "r").read(), height=750, scrolling=True)

#         # # Network
#         # # --- Build the graph ---
#         # G = nx.Graph()
#         # user_node = "USER_PAPER"
#         # G.add_node(user_node, size=40, color="red")

#         # for r in results:
#         #     paper_node = r["paper_title"]
#         #     G.add_node(paper_node, size=10, title=paper_node)
#         #     G.add_edge(user_node, paper_node, weight=r["score"])

#         # # --- Create PyVis Network ---
#         # net = Network(height="600px", width="100%", notebook=False)

#         # for n, d in G.nodes(data=True):
#         #     color = "red" if n == user_node else "lightblue"
#         #     net.add_node(n, label=str(n), value=d.get("size", 10), color=color, title=d.get("title", n))

#         # for u, v, d in G.edges(data=True):
#         #     net.add_edge(u, v, value=d.get("weight", 0.1)*10)

#         # # --- Save HTML and render in Streamlit ---
#         # with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#         #     net.write_html(tmp.name)
#         #     html = open(tmp.name, "r", encoding="utf-8").read()
#         #     st.components.v1.html(html, height=650, scrolling=True)

#         # CSV export
#         csv_data = "paper_title,journal,publisher,scopus_id,score,confidence\n"
#         for r in results:
#             csv_data += f"{r['paper_title']},{r['journal']},{r['publisher']},{r['scopus_id']},{r['score']},{r['confidence']}\n"
#         st.download_button("Download recommendations CSV", data=csv_data, file_name="recommendations.csv")
