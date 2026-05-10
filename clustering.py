import numpy as np
import pandas as pd
import joblib
import os
import re
from collections import Counter

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px

MODELS_DIR = "models_saved"
EMBEDDINGS_FILE = f"{MODELS_DIR}/embeddings.npy"
CLUSTER_FILE = f"{MODELS_DIR}/cluster_results.pkl"
ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_STOPWORDS = {
    "это", "как", "так", "все", "очень", "уже", "его", "что", "для", "при",
    "ещё", "ни", "но", "или", "себе", "нет", "есть", "был", "бы", "же",
    "мне", "мы", "вы", "он", "она", "они", "тут", "там", "чем", "кто",
    "этот", "эта", "эти", "тот", "та", "те", "сам", "раз", "два", "три",
    "после", "чтобы", "если", "только", "даже", "хотя", "тоже", "зато",
    "сразу", "буду", "быть", "можно", "надо", "нужно", "пока", "свой",
    "такой", "такие", "само",
}


def get_embeddings(texts: list[str], force: bool = False) -> np.ndarray:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(EMBEDDINGS_FILE) and not force:
        return np.load(EMBEDDINGS_FILE)
    from sentence_transformers import SentenceTransformer
    print("Computing sentence embeddings...")
    model = SentenceTransformer(ST_MODEL)
    embs = model.encode(texts, show_progress_bar=True, batch_size=32)
    np.save(EMBEDDINGS_FILE, embs)
    return embs


def cluster_reviews(df: pd.DataFrame, n_clusters: int = 5, force: bool = False) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(CLUSTER_FILE) and not force:
        return joblib.load(CLUSTER_FILE)

    texts = df["text"].tolist()
    embs = get_embeddings(texts, force)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embs)

    dbscan = DBSCAN(eps=0.5, min_samples=3, metric="cosine")
    dbscan_labels = dbscan.fit_predict(embs)

    pca = PCA(n_components=2, random_state=42)
    embs_2d = pca.fit_transform(embs)

    result = {
        "embeddings": embs,
        "embeddings_2d": embs_2d,
        "kmeans_labels": kmeans_labels,
        "dbscan_labels": dbscan_labels,
        "n_clusters": n_clusters,
        "df": df.reset_index(drop=True),
    }
    joblib.dump(result, CLUSTER_FILE)
    return result


def get_cluster_stats(result: dict) -> dict:
    """Top-10 words and average sentiment per KMeans cluster."""
    df = result["df"]
    labels = result["kmeans_labels"]
    stats = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        cdf = df[mask]
        words = []
        for text in cdf["text"]:
            words.extend(re.findall(r"[а-яё]{4,}", text.lower()))
        words = [w for w in words if w not in _STOPWORDS]
        top_words = [w for w, _ in Counter(words).most_common(10)]
        b = cdf[cdf["sentiment"].notna()]
        avg_sent = float(b["sentiment"].astype(float).mean()) if len(b) > 0 else 0.5
        stats[int(cid)] = {
            "top_words": top_words,
            "avg_sentiment": avg_sent,
            "size": int(mask.sum()),
        }
    return stats


def create_cluster_figure(result: dict) -> "go.Figure":
    df = result["df"]
    embs_2d = result["embeddings_2d"]
    labels = result["kmeans_labels"]

    plot_df = pd.DataFrame({
        "PC1": embs_2d[:, 0],
        "PC2": embs_2d[:, 1],
        "Кластер": labels.astype(str),
        "Текст": df["text"].str[:80],
        "Товар": df["product"],
        "Оценка": df["rating"],
        "Тональность": df["sentiment"].map({1.0: "Позитив", 0.0: "Негатив"}).fillna("Нейтрал"),
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Кластер",
        hover_data={"Текст": True, "Товар": True, "Оценка": True, "Тональность": True},
        title="Кластеры отзывов (PCA + KMeans)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.75))
    return fig
