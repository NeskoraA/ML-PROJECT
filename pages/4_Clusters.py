"""Страница визуализации кластеров (интерактивный plotly)."""
import uuid
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
from logger_setup import log

st.set_page_config(page_title="Кластеры", page_icon="🔵", layout="wide")
st.title("🔵 Кластеризация отзывов")

MODELS_DIR = "models_saved"
CLUSTER_FILE = f"{MODELS_DIR}/cluster_results.pkl"


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


@st.cache_resource(show_spinner=False)
def load_cluster_result():
    if not os.path.exists(CLUSTER_FILE):
        return None
    return joblib.load(CLUSTER_FILE)


user_id = get_user_id()
result = load_cluster_result()

if result is None:
    st.warning("Кластеры ещё не посчитаны. Откройте главную страницу — они сформируются автоматически.")
    st.stop()

from clustering import get_cluster_stats, create_cluster_figure

# ── Info ───────────────────────────────────────────────────────────────────────

df_cl = result["df"]
n_clusters = result["n_clusters"]
dbscan_labels = result["dbscan_labels"]
n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

c1, c2, c3 = st.columns(3)
c1.metric("Метод",        "KMeans + DBSCAN")
c2.metric("Кластеров KMeans", n_clusters)
c3.metric("Кластеров DBSCAN", n_dbscan)

# ── Scatter ────────────────────────────────────────────────────────────────────

st.subheader("🗺️ Интерактивная карта кластеров (PCA)")
fig = create_cluster_figure(result)
st.plotly_chart(fig, use_container_width=True)
st.caption("Наведите курсор на точку, чтобы увидеть текст отзыва.")

# ── Cluster stats ──────────────────────────────────────────────────────────────

st.subheader("📊 Статистика по кластерам")
stats = get_cluster_stats(result)

for cid, info in stats.items():
    with st.expander(
        f"Кластер {cid}  |  Отзывов: {info['size']}  |  "
        f"Средняя тональность: {info['avg_sentiment']:.0%}"
    ):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Топ-10 слов:**")
            st.write(", ".join(info["top_words"]) if info["top_words"] else "—")
        with c2:
            sent_pct = info["avg_sentiment"]
            color = "green" if sent_pct > 0.6 else ("red" if sent_pct < 0.4 else "orange")
            st.markdown(f"**Доля позитивных:** :{color}[{sent_pct:.0%}]")
            st.progress(sent_pct)

        st.markdown("**Примеры отзывов из кластера:**")
        mask = result["kmeans_labels"] == cid
        sample = df_cl[mask][["text", "product", "rating"]].sample(
            min(5, mask.sum()), random_state=42
        )
        st.dataframe(sample.rename(columns={
            "text": "Текст", "product": "Товар", "rating": "Оценка"
        }), hide_index=True, use_container_width=True)

# ── DBSCAN summary ─────────────────────────────────────────────────────────────

st.subheader("🔴 DBSCAN: выбросы")
noise_count = int((dbscan_labels == -1).sum())
st.info(
    f"DBSCAN обнаружил **{n_dbscan} кластеров** и **{noise_count} выбросов** "
    f"(шум, метка −1) из {len(dbscan_labels)} отзывов."
)

log(user_id, "visit_clusters")
