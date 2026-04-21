# 🗺️ Semantic News Navigator

A topological-style **semantic map of news articles** — articles are positioned by *meaning*, not keywords.
Similar topics cluster together automatically using sentence embeddings + UMAP + KMeans.

---

## ✨ Features

| Feature | Details |
|---|---|
| **News Fetching** | Live data via NewsAPI; graceful fallback to `sample_news.json` |
| **Semantic Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Dimensionality Reduction** | UMAP (preferred) → PCA fallback |
| **Clustering** | KMeans, k=3–10 (user-adjustable) |
| **Interactive Map** | Plotly scatter — hover, zoom, pan |
| **Semantic Search** | Cosine similarity; highlights top-N matches on map |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### 2. (Optional) Add your NewsAPI key

**Option A — Streamlit secrets (recommended):**

Edit `.streamlit/secrets.toml`:
```toml
NEWS_API_KEY = "your_api_key_here"
```

**Option B — Sidebar input:**
Paste your key directly in the sidebar text field at runtime.

> 🆓 Get a free key at [https://newsapi.org](https://newsapi.org)
>
> ⚠️ Without a key the app uses built-in sample data (30 articles across 6 categories).

### 3. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
semantic_news_navigator/
├── app.py               ← Main Streamlit application
├── embedder.py          ← Embeddings, UMAP/PCA, KMeans, semantic search
├── news_fetcher.py      ← NewsAPI fetching + fallback loader
├── sample_news.json     ← 30 curated sample articles (6 categories)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── .streamlit/
    ├── config.toml      ← Dark theme config
    └── secrets.toml     ← API key (add yours here)
```

---

## ⚙️ Sidebar Controls

| Control | Description |
|---|---|
| **API Key** | NewsAPI key (overrides secrets.toml) |
| **Category** | general / technology / business / sports / health / entertainment |
| **K (clusters)** | Number of KMeans clusters (3–10) |
| **UMAP neighbours** | Controls local vs global structure balance |
| **UMAP min_dist** | How tightly points are packed |
| **Fetch / Refresh** | Re-runs the full pipeline |

---

## 🔍 Semantic Search

Type any phrase in the search bar (e.g. *"artificial intelligence"*, *"climate change"*, *"stock market"*).

- Articles are ranked by **cosine similarity** with your query embedding
- Top matches are **highlighted on the map** with a red ring + larger dot
- Matched articles appear in a ranked list below the chart with similarity scores

---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[sentence-transformers](https://www.sbert.net)** — `all-MiniLM-L6-v2` embeddings
- **[umap-learn](https://umap-learn.readthedocs.io)** — Dimensionality reduction
- **[scikit-learn](https://scikit-learn.org)** — KMeans, PCA, cosine similarity
- **[Plotly](https://plotly.com)** — Interactive scatter plot
- **[NewsAPI](https://newsapi.org)** — Live news headlines

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'umap'`**
```bash
pip install umap-learn
```
The app automatically falls back to PCA if UMAP is unavailable.

**App shows sample data even with a valid API key**
- Verify the key at [https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_KEY](https://newsapi.org)
- Free plan only allows requests from localhost (developer mode)

**Slow first load**
The sentence-transformer model (~90 MB) downloads on first run and is cached.
Subsequent runs are fast.
