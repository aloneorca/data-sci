import streamlit as st
from recommend import recommend_for_text
import pandas as pd
import altair as alt   # for scatter plot

st.set_page_config(layout="wide", page_title="Scopus Recommender")
st.title("📘 Scopus Journal Recommender")


# Sidebar inputs

with st.sidebar:
    st.header("New Manuscript")
    title = st.text_input("Title", "")
    abstract = st.text_area("Abstract", "", height=200)
    journal_or_paper = st.selectbox("Recommend at:", ["Journal Level", "Paper Level"])
    paper_level = journal_or_paper == "Paper Level"
    use_classifier = journal_or_paper == "Journal Level"
    top_k = st.slider("Top K results", 5, 30, 15)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Main 

if st.sidebar.button("Recommend"):
    if not title.strip():
        st.error("Please provide a title.")
        st.stop()

    text = title.strip() + ". " + (abstract.strip() if abstract.strip().lower() != "not available" else "")

    with st.spinner("Computing recommendations..."):
        results = recommend_for_text(
            text,
            model_name=model_name,
            top_k=top_k,
            use_classifier=use_classifier,
            paper_level=paper_level
        )

    # convert to dataframe    
    rows = []
    for r in results:
        rows.append({
            "paper_title": r["paper_title"],
            "journal": r["journal"],
            "publisher": r["publisher"],
            "scopus_id": r["scopus_id"],
            "score": float(r["score"]),
            "confidence": r["confidence"]
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True).reset_index()


    # Summary Section
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Journal", df.iloc[0]["journal"])
    col2.metric("Highest Score", f"{df['score'].max():.3f}")
    col3.metric("Total Recommendations", len(df))

    # Scatter plot of scores
    st.subheader("📈 Score Distribution")

    scatter = (
        alt.Chart(df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("index:Q", title="Rank"),
            y=alt.Y("score:Q", title="Score"),
            tooltip=["paper_title", "journal", "score", "confidence"]
        )
    )

    st.altair_chart(scatter, use_container_width=True)

    # Group by confidence
    st.subheader("📌 Recommendations by Confidence")

    confidence_order = ["HIGH", "MEDIUM", "LOW"]

    for level in confidence_order:
        subset = df[df["confidence"] == level]
        if subset.empty:
            continue

        st.markdown(f"### {level} Confidence ({len(subset)})")

        for _, row in subset.iterrows():
            with st.container(border=True):
                st.markdown(f"**📄 Paper Title:** {row['paper_title']}")
                st.markdown(f"**📰 Journal:** {row['journal']}")
                st.markdown(f"**🏢 Publisher:** {row['publisher']}")
                st.markdown(f"**🆔 Scopus ID:** {row['scopus_id']}")
                st.markdown(f"**⭐ Score:** {row['score']:.4f}")
                st.markdown(f"**🔎 Confidence:** {row['confidence']}")

                st.markdown("---")

    # CSV Download
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv"
    )
