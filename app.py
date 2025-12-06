import streamlit as st
from recommend import recommend_for_text
import pandas as pd
import altair as alt
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Scopus Recommender",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic improvements (Dark Mode Compatible)
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(28, 131, 225, 0.3);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .paper-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    .paper-card h4 {
        color: #ffffff;
        margin-top: 0;
    }
    .paper-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
    }
    .confidence-high {
        border-left-color: #4CAF50 !important;
        background-color: rgba(76, 175, 80, 0.08) !important;
    }
    .confidence-medium {
        border-left-color: #FFC107 !important;
        background-color: rgba(255, 193, 7, 0.08) !important;
    }
    .confidence-low {
        border-left-color: #FF5722 !important;
        background-color: rgba(255, 87, 34, 0.08) !important;
    }
    h1 {
        color: #4da6ff;
        font-weight: 600;
    }
    h2 {
        color: #66b3ff;
        font-weight: 500;
        margin-top: 2rem;
    }
    h3 {
        color: #80bfff;
    }
    /* Better visibility for expanders in dark mode */
    .streamlit-expanderHeader {
        background-color: rgba(28, 131, 225, 0.1) !important;
        border: 1px solid rgba(28, 131, 225, 0.3) !important;
    }
    /* Improve dataframe visibility */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Header
st.title("📚 Scopus Journal Recommender")
st.markdown("*Find the perfect journal for your research paper*")
st.divider()

# Sidebar - Input Section
with st.sidebar:
    st.header("🔍 Search Parameters")
    
    # Input fields
    title = st.text_input("📝 Paper Title", placeholder="Enter your paper title...")
    abstract = st.text_area("📄 Abstract", placeholder="Enter your abstract...", height=200)
    
    st.divider()
    
    # Recommendation settings
    st.subheader("⚙️ Settings")
    journal_or_paper = st.selectbox(
        "Recommendation Level:",
        ["Journal Level", "Paper Level"],
        help="Choose whether to recommend journals or specific papers"
    )
    paper_level = journal_or_paper == "Paper Level"
    use_classifier = journal_or_paper == "Journal Level"
    
    top_k = st.slider("Number of Results", 5, 30, 15, help="Number of recommendations to display")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    st.divider()
    
    # Recommend button
    recommend_btn = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
    
    st.divider()
    
    # Search History
    st.subheader("📜 Recent Searches")
    if st.session_state.search_history:
        for idx, search in enumerate(reversed(st.session_state.search_history[-5:])):
            with st.expander(f"{search['timestamp']} - {search['title'][:30]}..."):
                st.write(f"**Level:** {search['level']}")
                st.write(f"**Results:** {search['count']}")
                if st.button(f"Load", key=f"load_{len(st.session_state.search_history)-idx-1}"):
                    st.session_state.current_results = search['results']
                    st.rerun()
    else:
        st.info("No recent searches")

# Main logic
if recommend_btn:
    if not title.strip():
        st.error("⚠️ Please provide a paper title.")
        st.stop()

    text = title.strip() + ". " + (abstract.strip() if abstract.strip().lower() != "not available" else "")

    with st.spinner("🔄 Computing recommendations..."):
        results = recommend_for_text(
            text,
            model_name=model_name,
            top_k=top_k,
            use_classifier=use_classifier,
            paper_level=paper_level
        )
    
    # Store results and add to history
    st.session_state.current_results = results
    st.session_state.search_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'title': title,
        'level': journal_or_paper,
        'count': len(results),
        'results': results
    })
    
    st.success("✅ Recommendations generated successfully!")

# Display results if available
if st.session_state.current_results:
    results = st.session_state.current_results
    
    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append({
            "paper_title": r["paper_title"],
            "journal": r["journal"],
            "publisher": r["publisher"],
            "scopus_id": r["scopus_id"],
            "score": float(r["score"]),
            "confidence": r["confidence"],
            "confidence_num": r.get("confidence_num", 0)
        })
    df = pd.DataFrame(rows)
    
    # Summary Section
    st.header("📊 Recommendation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Top Journal", df.iloc[0]["journal"])
    with col2:
        st.metric("⭐ Highest Score", f"{df['score'].max():.3f}")
    with col3:
        st.metric("📈 Total Results", len(df))
    with col4:
        high_conf = len(df[df["confidence"] == "HIGH"])
        st.metric("🎯 High Confidence", high_conf)
    
    st.divider()
    
    # Sorting and filtering options
    st.header("🎯 Results")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Score (High to Low)", "Score (Low to High)", "Confidence", "Journal Name", "Publisher"]
        )
    with col2:
        filter_conf = st.multiselect(
            "Filter by Confidence:",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )
    
    # Apply sorting
    if sort_by == "Score (High to Low)":
        df_sorted = df.sort_values("score", ascending=False)
    elif sort_by == "Score (Low to High)":
        df_sorted = df.sort_values("score", ascending=True)
    elif sort_by == "Confidence":
        df_sorted = df.sort_values("confidence_num", ascending=False)
    elif sort_by == "Journal Name":
        df_sorted = df.sort_values("journal")
    else:  # Publisher
        df_sorted = df.sort_values("publisher")
    
    # Apply filtering
    df_filtered = df_sorted[df_sorted["confidence"].isin(filter_conf)].reset_index(drop=True)
    
    # Visualization Section
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Score Distribution", "🎨 Confidence Breakdown", "📋 Data Table", "🔗 Network View"])
    
    with tab1:
        st.subheader("Score Distribution Across Recommendations")
        
        # Enhanced scatter plot
        scatter = (
            alt.Chart(df_filtered.reset_index())
            .mark_circle(size=120, opacity=0.7)
            .encode(
                x=alt.X("index:Q", title="Rank", scale=alt.Scale(domain=[0, len(df_filtered)])),
                y=alt.Y("score:Q", title="Similarity Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("confidence:N", 
                    scale=alt.Scale(
                        domain=["HIGH", "MEDIUM", "LOW"],
                        range=["#4CAF50", "#FFC107", "#FF5722"]
                    ),
                    legend=alt.Legend(title="Confidence Level")
                ),
                tooltip=[
                    alt.Tooltip("paper_title:N", title="Paper"),
                    alt.Tooltip("journal:N", title="Journal"),
                    alt.Tooltip("score:Q", title="Score", format=".4f"),
                    alt.Tooltip("confidence:N", title="Confidence")
                ]
            )
            .properties(height=400)
            .interactive()
        )
        
        st.altair_chart(scatter, use_container_width=True)
        
        # Score histogram
        st.subheader("Score Frequency Distribution")
        hist = (
            alt.Chart(df_filtered)
            .mark_bar(opacity=0.7, color="#1f77b4")
            .encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Score Range"),
                y=alt.Y("count()", title="Frequency"),
                tooltip=["count()"]
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)
    
    with tab2:
        st.subheader("Recommendations by Confidence Level")
        
        # Confidence distribution pie chart
        conf_counts = df_filtered["confidence"].value_counts().reset_index()
        conf_counts.columns = ["confidence", "count"]
        
        pie = (
            alt.Chart(conf_counts)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("confidence:N",
                    scale=alt.Scale(
                        domain=["HIGH", "MEDIUM", "LOW"],
                        range=["#4CAF50", "#FFC107", "#FF5722"]
                    ),
                    legend=alt.Legend(title="Confidence Level")
                ),
                tooltip=["confidence", "count"]
            )
            .properties(height=400)
        )
        
        st.altair_chart(pie, use_container_width=True)
        
        # Bar chart by confidence
        st.subheader("Average Score by Confidence Level")
        avg_scores = df_filtered.groupby("confidence")["score"].mean().reset_index()
        
        bar = (
            alt.Chart(avg_scores)
            .mark_bar()
            .encode(
                x=alt.X("confidence:N", title="Confidence Level", sort=["HIGH", "MEDIUM", "LOW"]),
                y=alt.Y("score:Q", title="Average Score"),
                color=alt.Color("confidence:N",
                    scale=alt.Scale(
                        domain=["HIGH", "MEDIUM", "LOW"],
                        range=["#4CAF50", "#FFC107", "#FF5722"]
                    ),
                    legend=None
                ),
                tooltip=["confidence", alt.Tooltip("score:Q", format=".4f")]
            )
            .properties(height=300)
        )
        
        st.altair_chart(bar, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Results Table")
        
        # Display dataframe with formatting
        display_df = df_filtered[["paper_title", "journal", "publisher", "score", "confidence", "scopus_id"]].copy()
        display_df["score"] = display_df["score"].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "paper_title": st.column_config.TextColumn("Paper Title", width="large"),
                "journal": st.column_config.TextColumn("Journal", width="medium"),
                "publisher": st.column_config.TextColumn("Publisher", width="medium"),
                "score": st.column_config.TextColumn("Score", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "scopus_id": st.column_config.TextColumn("Scopus ID", width="small")
            }
        )
        
        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name=f"scopus_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Journal Distribution Network")
        
        # Top journals bar chart
        top_journals = df_filtered["journal"].value_counts().head(10).reset_index()
        top_journals.columns = ["journal", "count"]
        
        journal_bar = (
            alt.Chart(top_journals)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("count:Q", title="Number of Recommendations"),
                y=alt.Y("journal:N", title="Journal", sort="-x"),
                tooltip=["journal", "count"]
            )
            .properties(height=400)
        )
        
        st.altair_chart(journal_bar, use_container_width=True)
        
        # Publisher distribution
        st.subheader("Top Publishers")
        top_publishers = df_filtered["publisher"].value_counts().head(8).reset_index()
        top_publishers.columns = ["publisher", "count"]
        
        pub_bar = (
            alt.Chart(top_publishers)
            .mark_bar(color="#ff7f0e")
            .encode(
                x=alt.X("count:Q", title="Number of Recommendations"),
                y=alt.Y("publisher:N", title="Publisher", sort="-x"),
                tooltip=["publisher", "count"]
            )
            .properties(height=350)
        )
        
        st.altair_chart(pub_bar, use_container_width=True)
    
    st.divider()
    
    # Detailed Results by Confidence
    st.header("📌 Detailed Recommendations")
    
    confidence_order = ["HIGH", "MEDIUM", "LOW"]
    confidence_colors = {
        "HIGH": "confidence-high",
        "MEDIUM": "confidence-medium",
        "LOW": "confidence-low"
    }
    confidence_icons = {
        "HIGH": "🟢",
        "MEDIUM": "🟡",
        "LOW": "🔴"
    }
    
    for level in confidence_order:
        if level not in filter_conf:
            continue
            
        subset = df_filtered[df_filtered["confidence"] == level]
        if subset.empty:
            continue
        
        with st.expander(f"{confidence_icons[level]} {level} Confidence ({len(subset)} results)", expanded=(level == "HIGH")):
            for idx, row in subset.iterrows():
                st.markdown(f"""
                    <div class="paper-card {confidence_colors[level]}">
                        <h4>📄 {row['paper_title']}</h4>
                        <p><strong>📰 Journal:</strong> {row['journal']}</p>
                        <p><strong>🏢 Publisher:</strong> {row['publisher']}</p>
                        <p><strong>🆔 Scopus ID:</strong> {row['scopus_id']}</p>
                        <p><strong>⭐ Similarity Score:</strong> {row['score']:.4f}</p>
                        <p><strong>🎯 Confidence:</strong> {confidence_icons[level]} {row['confidence']}</p>
                    </div>
                """, unsafe_allow_html=True)

else:
    # Welcome screen when no results
    st.info("👈 Enter your paper details in the sidebar and click **Get Recommendations** to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - Smart journal recommendations
        - Paper-level matching
        - Confidence scoring
        - Interactive visualizations
        """)
    
    with col2:
        st.markdown("### 📊 Analytics")
        st.markdown("""
        - Score distribution charts
        - Confidence breakdowns
        - Publisher insights
        - Sortable results
        """)
    
    with col3:
        st.markdown("### 💾 History")
        st.markdown("""
        - Recent search tracking
        - Quick result reload
        - CSV export
        - Customizable filters
        """)

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>📚 Scopus Journal Recommender | Powered by Sentence Transformers</p>",
    unsafe_allow_html=True
)
