"""Страница визуализации кластеров — клик на кластер показывает примеры."""
import uuid
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
from logger_setup import log

st.set_page_config(page_title="Кластеры", layout="wide")
st.title("Кластеризация отзывов")

MODELS_DIR = "models_saved"
CLUSTER_FILE = f"{MODELS_DIR}/cluster_results.pkl"


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


@st.cache_resource(show_spinner=False)
def load_result():
    if not os.path.exists(CLUSTER_FILE):
        return None
    return joblib.load(CLUSTER_FILE)


user_id = get_user_id()
result = load_result()

if result is None:
    st.warning("Кластеры ещё не посчитаны. Откройте главную страницу — они сформируются автоматически.")
    st.stop()

from clustering import get_cluster_stats, create_cluster_figure

df_cl   = result["df"]
labels  = result["kmeans_labels"]
n_cl    = result["n_clusters"]
db_lab  = result["dbscan_labels"]
n_dbscan = len(set(db_lab)) - (1 if -1 in db_lab else 0)
stats   = get_cluster_stats(result)

# ── Metrics ────────────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
c1.metric("Метод векторизации", "SentenceTransformer")
c2.metric("Кластеров KMeans",  n_cl)
c3.metric("Кластеров DBSCAN",  n_dbscan)

# ── Interactive scatter ────────────────────────────────────────────────────────

st.subheader("Интерактивная карта кластеров (PCA + KMeans)")
st.caption("Нажмите на точку или выделите область — ниже появятся примеры отзывов из этого кластера.")

fig = create_cluster_figure(result)

# on_select="rerun" перезапускает страницу при клике и передаёт выбранные точки
event = st.plotly_chart(
    fig,
    key="cluster_chart",
    on_select="rerun",
    use_container_width=True,
)

# ── Show examples for clicked cluster ─────────────────────────────────────────

selected_cluster = None

if event and event.selection and event.selection.points:
    # curve_number: порядковый номер трассы в plotly px.scatter с color=
    # каждый кластер — отдельная трасса, отсортированная по значению
    curve = event.selection.points[0].get("curve_number", 0)
    # Сопоставляем curve_number → метку кластера
    unique_clusters = sorted(set(str(c) for c in labels))
    if curve < len(unique_clusters):
        selected_cluster = int(unique_clusters[curve])

    pt_idx = event.selection.points[0].get("point_index")
    if pt_idx is not None and pt_idx < len(labels):
        selected_cluster = int(labels[pt_idx])

if selected_cluster is not None:
    info = stats.get(selected_cluster, {})
    st.success(f"Выбран **Кластер {selected_cluster}** — {info.get('size', '?')} отзывов")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Топ-10 ключевых слов:**")
        words = info.get("top_words", [])
        st.write(", ".join(words) if words else "—")
    with cc2:
        pct = info.get("avg_sentiment", 0.5)
        color = "green" if pct > 0.6 else ("red" if pct < 0.4 else "orange")
        st.markdown(f"**Доля позитивных отзывов:** :{color}[{pct:.0%}]")
        st.progress(float(pct))

    st.markdown("**Примеры отзывов из кластера:**")
    mask = labels == selected_cluster
    sample = df_cl[mask][["text", "product", "rating"]].sample(
        min(5, int(mask.sum())), random_state=42
    ).rename(columns={"text": "Текст", "product": "Товар", "rating": "Оценка"})
    st.dataframe(sample, hide_index=True, use_container_width=True)
    st.divider()

# ── Cluster overview table ─────────────────────────────────────────────────────

st.subheader("Статистика всех кластеров")

for cid, info in stats.items():
    with st.expander(
        f"Кластер {cid}  |  Отзывов: {info['size']}  |  "
        f"Тональность: {info['avg_sentiment']:.0%} позитивных"
    ):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Топ-10 слов:**")
            st.write(", ".join(info["top_words"]) if info["top_words"] else "—")
        with cc2:
            pct = info["avg_sentiment"]
            color = "green" if pct > 0.6 else ("red" if pct < 0.4 else "orange")
            st.markdown(f"**Доля позитивных:** :{color}[{pct:.0%}]")
            st.progress(float(pct))
        st.markdown("**Примеры:**")
        mask = labels == cid
        sample = df_cl[mask][["text", "product", "rating"]].sample(
            min(5, int(mask.sum())), random_state=cid
        ).rename(columns={"text": "Текст", "product": "Товар", "rating": "Оценка"})
        st.dataframe(sample, hide_index=True, use_container_width=True)

# ── DBSCAN ────────────────────────────────────────────────────────────────────

st.subheader("DBSCAN: результат")
noise = int((db_lab == -1).sum())
st.info(
    f"DBSCAN нашёл **{n_dbscan} кластеров** и **{noise} выбросов** (шум) "
    f"из {len(db_lab)} отзывов."
)

log(user_id, "visit_clusters")
