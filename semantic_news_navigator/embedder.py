"""
embedder.py  —  Sentence embeddings, dimensionality reduction, and clustering.
"""

import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# Model loading (cached across reruns)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading sentence-transformer model…")
def load_model():
    """Load and cache the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Computing semantic embeddings…")
def compute_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    """
    Embed a tuple of strings.
    Using tuple instead of list so Streamlit can hash it.
    """
    model = load_model()
    return model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)


def articles_to_texts(articles: list[dict]) -> list[str]:
    """Combine title + description for each article into a single string."""
    return [
        f"{a['title']}. {a.get('description', '')}".strip()
        for a in articles
    ]


# ──────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────

def reduce_to_2d(embeddings: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.3) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D.
    Tries UMAP first; falls back to PCA.
    Returns (coords_2d, method_used_str)
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(embeddings) - 1),
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        return coords, "UMAP"
    except Exception:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        return coords, "PCA"


# ──────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────

def cluster_articles(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Run KMeans on embeddings; returns integer cluster labels."""
    k = min(k, len(embeddings))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(embeddings)


def extract_cluster_keywords(articles: list[dict], labels: np.ndarray, k: int) -> dict[int, str]:
    """
    For each cluster, find the top 2-3 meaningful words from titles.
    Returns {cluster_id: "keyword1 · keyword2"}.
    """
    import re
    from collections import Counter

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "has", "have", "had", "its", "it", "this",
        "that", "they", "their", "he", "she", "his", "her", "we", "you",
        "new", "say", "says", "said", "after", "over", "than", "more",
        "up", "out", "about", "into", "what", "who", "how", "when",
        "will", "can", "not", "no", "us", "amid", "amid", "amid",
    }

    cluster_labels_map = {}
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        words = []
        for i in idxs:
            title = articles[i].get("title", "")
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", title.lower())
            words.extend(t for t in tokens if t not in STOPWORDS)
        top = [w for w, _ in Counter(words).most_common(3)]
        cluster_labels_map[cid] = " · ".join(top) if top else f"Cluster {cid + 1}"

    return cluster_labels_map


# ──────────────────────────────────────────────
# Semantic search
# ──────────────────────────────────────────────

def semantic_search(query: str, embeddings: np.ndarray, top_n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed query and return (indices, similarity_scores) of top-N articles.
    """
    model = load_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_n]
    return top_idx, scores[top_idx]
