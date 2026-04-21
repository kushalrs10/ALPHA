"""
app.py  —  Semantic News Navigator
Topological-style semantic map of news articles.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embedder import (
    articles_to_texts,
    cluster_articles,
    compute_embeddings,
    extract_cluster_keywords,
    reduce_to_2d,
    semantic_search,
)
from news_fetcher import CATEGORY_OPTIONS, fetch_news


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic News Navigator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f0f1a; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2d2d4e;
    }

    /* Title gradient */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Search bar */
    .search-container {
        background: linear-gradient(135deg, #1e1e3f, #1a2540);
        border: 1px solid #3d3d6b;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
    }

    /* Article card */
    .article-card {
        background: linear-gradient(135deg, #1a1a30, #1e2040);
        border: 1px solid #2d2d50;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        transition: border-color 0.2s;
    }
    .article-card:hover { border-color: #6366f1; }
    .article-title { color: #e2e8f0; font-weight: 600; font-size: 0.95rem; }
    .article-meta  { color: #64748b; font-size: 0.78rem; margin-top: 0.2rem; }
    .article-desc  { color: #94a3b8; font-size: 0.85rem; margin-top: 0.4rem; }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
        margin-left: 0.5rem;
    }

    /* Stat chips */
    .stat-row { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .stat-chip {
        background: #1e1e3f;
        border: 1px solid #3d3d6b;
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        font-size: 0.82rem;
        color: #a5b4fc;
    }
    .stat-chip span { color: #e2e8f0; font-weight: 700; }

    /* Plotly container */
    .plot-container { border-radius: 14px; overflow: hidden; }

    /* Badges */
    .badge-umap { color: #34d399; font-size: 0.78rem; }
    .badge-pca  { color: #fbbf24; font-size: 0.78rem; }

    /* Divider */
    hr { border-color: #2d2d4e; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Sidebar  —  Controls
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🗺️ Navigator Controls")
    st.markdown("---")

    # ── API Key ──────────────────────────────
    st.markdown("### 🔑 NewsAPI Key")
    try:
        api_key_default = st.secrets.get("NEWS_API_KEY", "")
    except Exception:
        api_key_default = ""

    api_key = st.text_input(
        "API Key",
        value=api_key_default,
        type="password",
        placeholder="Paste your NewsAPI key…",
        help="Get a free key at https://newsapi.org  —  leave blank to use sample data.",
    )

    # ── Category ─────────────────────────────
    st.markdown("### 📰 News Category")
    category = st.selectbox(
        "Category",
        options=CATEGORY_OPTIONS,
        index=0,
        format_func=lambda x: x.capitalize(),
    )

    # ── Clustering ───────────────────────────
    st.markdown("### 🔵 Clustering")
    n_clusters = st.slider("Number of clusters (K)", min_value=3, max_value=10, value=5)

    # ── Reduction ────────────────────────────
    st.markdown("### 📐 Dimensionality Reduction")
    n_neighbors = st.slider("UMAP neighbours", min_value=5, max_value=30, value=10)
    min_dist = st.slider("UMAP min_dist", min_value=0.05, max_value=0.9, value=0.3, step=0.05)

    # ── Fetch button ─────────────────────────
    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch / Refresh News", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Built with Streamlit · sentence-transformers · UMAP · Plotly</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "articles" not in st.session_state:
    st.session_state.articles = []
    st.session_state.embeddings = None
    st.session_state.coords = None
    st.session_state.labels = None
    st.session_state.reduction_method = ""
    st.session_state.cluster_keywords = {}
    st.session_state.used_fallback = False


# ─────────────────────────────────────────────
# Data pipeline  (fetch + embed + cluster)
# ─────────────────────────────────────────────

def run_pipeline(api_key, category, n_clusters, n_neighbors, min_dist):
    """Full data pipeline: fetch → embed → reduce → cluster."""
    with st.spinner("Fetching news articles…"):
        articles, used_fallback = fetch_news(api_key, category=category, page_size=50)

    if used_fallback:
        st.warning(
            "⚠️  Could not reach NewsAPI (missing/invalid key or network error). "
            "Showing **sample data** instead.  "
            "Add your free key at [newsapi.org](https://newsapi.org) to get live articles.",
            icon="⚠️",
        )

    if not articles:
        st.error("No articles found. Try a different category or check your API key.")
        return

    texts = articles_to_texts(articles)
    embeddings = compute_embeddings(tuple(texts))

    coords, method = reduce_to_2d(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
    labels = cluster_articles(embeddings, k=n_clusters)
    cluster_kws = extract_cluster_keywords(articles, labels, k=n_clusters)

    st.session_state.articles = articles
    st.session_state.embeddings = embeddings
    st.session_state.coords = coords
    st.session_state.labels = labels
    st.session_state.reduction_method = method
    st.session_state.cluster_keywords = cluster_kws
    st.session_state.used_fallback = used_fallback


# Auto-run on first load
if not st.session_state.articles or fetch_btn:
    run_pipeline(api_key, category, n_clusters, n_neighbors, min_dist)


# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────

articles = st.session_state.articles
embeddings = st.session_state.embeddings
coords = st.session_state.coords
labels = st.session_state.labels
cluster_kws = st.session_state.cluster_keywords
method = st.session_state.reduction_method

if not articles or coords is None:
    st.stop()


# ─── Hero ─────────────────────────────────────
st.markdown('<div class="hero-title">🗺️ Semantic News Navigator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Articles positioned by <b>meaning</b>, not keywords  —  similar topics cluster together.</div>',
    unsafe_allow_html=True,
)

# ─── Stats bar ────────────────────────────────
method_badge = (
    f'<span class="badge-umap">● UMAP</span>'
    if method == "UMAP"
    else f'<span class="badge-pca">● PCA</span>'
)
st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-chip">Articles <span>{len(articles)}</span></div>
        <div class="stat-chip">Clusters <span>{n_clusters}</span></div>
        <div class="stat-chip">Reduction {method_badge}</div>
        <div class="stat-chip">Category <span>{category.capitalize()}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Semantic Search bar
# ─────────────────────────────────────────────

st.markdown('<div class="search-container">', unsafe_allow_html=True)
col_s1, col_s2 = st.columns([5, 1])
with col_s1:
    search_query = st.text_input(
        "🔍 Semantic Search",
        placeholder='Search by meaning, e.g. "AI breakthrough" or "stock market crash"…',
        label_visibility="collapsed",
    )
with col_s2:
    top_n = st.number_input("Top N", min_value=1, max_value=20, value=5, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# Run search
search_indices = np.array([], dtype=int)
search_scores = np.array([])
if search_query.strip():
    with st.spinner("Computing semantic similarity…"):
        search_indices, search_scores = semantic_search(search_query, embeddings, top_n=int(top_n))


# ─────────────────────────────────────────────
# Build DataFrame for Plotly
# ─────────────────────────────────────────────

df = pd.DataFrame(
    {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "title": [a["title"] for a in articles],
        "description": [
            (a.get("description") or "")[:150] + ("…" if len(a.get("description") or "") > 150 else "")
            for a in articles
        ],
        "source": [a.get("source", "Unknown") for a in articles],
        "cluster_id": labels,
        "cluster_label": [cluster_kws.get(int(l), f"Cluster {l+1}") for l in labels],
        "is_match": [i in search_indices for i in range(len(articles))],
    }
)

# Marker sizes & borders
df["marker_size"] = df["is_match"].apply(lambda m: 18 if m else 9)
df["marker_line_width"] = df["is_match"].apply(lambda m: 3 if m else 0.5)
df["marker_line_color"] = df["is_match"].apply(lambda m: "#ff4444" if m else "rgba(255,255,255,0.15)")


# ─────────────────────────────────────────────
# Plotly scatter
# ─────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#6366f1", "#22d3ee", "#f59e0b", "#10b981",
    "#f43f5e", "#a855f7", "#3b82f6", "#84cc16",
    "#fb923c", "#e879f9",
]

fig = go.Figure()

# Draw one trace per cluster (enables legend toggle)
for cid in sorted(df["cluster_id"].unique()):
    mask = df["cluster_id"] == cid
    sub = df[mask]
    color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
    cluster_name = cluster_kws.get(int(cid), f"Cluster {cid + 1}")

    fig.add_trace(
        go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=cluster_name,
            marker=dict(
                color=color,
                size=sub["marker_size"],
                opacity=0.85,
                line=dict(
                    width=sub["marker_line_width"].tolist(),
                    color=sub["marker_line_color"].tolist(),
                ),
            ),
            customdata=np.stack(
                [sub["title"], sub["description"], sub["source"], sub["cluster_label"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<span style='color:#94a3b8'>%{customdata[2]}</span><br><br>"
                "%{customdata[1]}<br>"
                "<i>Cluster: %{customdata[3]}</i>"
                "<extra></extra>"
            ),
        )
    )

# Highlight search matches with a red ring overlay
if len(search_indices) > 0:
    match_df = df.iloc[search_indices]
    fig.add_trace(
        go.Scatter(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            name="🔍 Search Match",
            marker=dict(
                color="rgba(0,0,0,0)",
                size=26,
                line=dict(width=3, color="#ff4444"),
            ),
            customdata=np.stack(
                [match_df["title"], match_df["description"], match_df["source"]], axis=-1
            ),
            hovertemplate=(
                "<b>🔍 MATCH</b><br>"
                "<b>%{customdata[0]}</b><br>"
                "<span style='color:#94a3b8'>%{customdata[2]}</span><br><br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

fig.update_layout(
    height=650,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#10101e",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(20,20,40,0.85)",
        bordercolor="#3d3d6b",
        borderwidth=1,
        font=dict(size=11),
        itemsizing="constant",
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#1e1e35",
        zeroline=False,
        showticklabels=False,
        title="",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1e1e35",
        zeroline=False,
        showticklabels=False,
        title="",
    ),
    hoverlabel=dict(
        bgcolor="#1e1e3f",
        bordercolor="#6366f1",
        font=dict(color="#e2e8f0", size=12),
        align="left",
    ),
    title=dict(
        text="Semantic Article Map  —  hover to explore  |  scroll to zoom  |  drag to pan",
        font=dict(size=13, color="#64748b"),
        x=0.5,
        xanchor="center",
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})


# ─────────────────────────────────────────────
# Search results panel
# ─────────────────────────────────────────────

if search_query.strip() and len(search_indices) > 0:
    st.markdown(f"### 🔍 Top {len(search_indices)} semantic matches for *\"{search_query}\"*")
    for rank, (idx, score) in enumerate(zip(search_indices, search_scores), 1):
        art = articles[idx]
        cname = cluster_kws.get(int(labels[idx]), f"Cluster {labels[idx]+1}")
        desc = (art.get("description") or "")
        desc_display = desc[:200] + "…" if len(desc) > 200 else desc
        st.markdown(
            f"""
            <div class="article-card">
                <div class="article-title">
                    #{rank}
                    <a href="{art.get('url','#')}" target="_blank" style="color:#a5b4fc;text-decoration:none">
                        {art['title']}
                    </a>
                    <span class="score-badge">{score:.2%}</span>
                </div>
                <div class="article-meta">
                    📰 {art.get('source','Unknown')} &nbsp;·&nbsp;
                    🏷️ {cname}
                </div>
                <div class="article-desc">{desc_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")


# ─────────────────────────────────────────────
# All articles grid (expandable)
# ─────────────────────────────────────────────

with st.expander(f"📋 Browse all {len(articles)} articles", expanded=False):
    cols = st.columns(2)
    for i, art in enumerate(articles):
        cid = int(labels[i])
        cname = cluster_kws.get(cid, f"Cluster {cid+1}")
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        desc = (art.get("description") or "")[:140]
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="article-card" style="border-left: 3px solid {color}">
                    <div class="article-title">
                        <a href="{art.get('url','#')}" target="_blank"
                           style="color:#e2e8f0;text-decoration:none">{art['title']}</a>
                    </div>
                    <div class="article-meta">
                        📰 {art.get('source','?')} &nbsp;·&nbsp;
                        <span style="color:{color}">● {cname}</span>
                    </div>
                    <div class="article-desc">{desc}{"…" if len(art.get("description",""))>140 else ""}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
