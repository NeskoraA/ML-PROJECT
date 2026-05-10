"""Страница статистики по товарам."""
import uuid
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from logger_setup import log

st.set_page_config(page_title="Статистика товаров", page_icon="📦", layout="wide")
st.title("📦 Статистика по товарам")


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


@st.cache_data(ttl=30, show_spinner=False)
def load_data() -> pd.DataFrame:
    from database import get_reviews
    return get_reviews()


user_id = get_user_id()
df = load_data()
df_b = df[df["sentiment"].notna()].copy()
df_b["sentiment"] = df_b["sentiment"].astype(float)

# ── Summary table ──────────────────────────────────────────────────────────────

stats = df.groupby("product").agg(
    avg_rating=("rating", "mean"),
    review_count=("rating", "count"),
).reset_index()

pos = df_b.groupby("product")["sentiment"].mean().reset_index()
pos.columns = ["product", "pos_ratio"]
stats = stats.merge(pos, on="product", how="left")
stats["pos_ratio"] = stats["pos_ratio"].fillna(0)

display = stats.copy()
display["avg_rating"]    = display["avg_rating"].round(2)
display["pos_ratio"]     = (display["pos_ratio"] * 100).round(1).astype(str) + "%"
display.columns = ["Товар", "Средний рейтинг", "Кол-во отзывов", "% позитивных"]

st.subheader("📋 Сводная таблица по товарам")
st.dataframe(display, use_container_width=True, hide_index=True)

# Highlight best product
best = stats.loc[stats["avg_rating"].idxmax()]
st.success(f"Лучший товар по рейтингу: **{best['product']}** — {best['avg_rating']:.2f} ⭐")

# ── Bar charts ─────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1:
    st.subheader("Средний рейтинг")
    fig = px.bar(
        stats.sort_values("avg_rating"),
        x="avg_rating", y="product", orientation="h",
        color="avg_rating", color_continuous_scale="RdYlGn",
        labels={"avg_rating": "Средний рейтинг", "product": "Товар"},
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Доля позитивных отзывов")
    fig = px.bar(
        stats.sort_values("pos_ratio"),
        x="pos_ratio", y="product", orientation="h",
        color="pos_ratio", color_continuous_scale="RdYlGn",
        labels={"pos_ratio": "Доля позитивных", "product": "Товар"},
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Top-5 negative words per product ──────────────────────────────────────────

from utils import top_negative_words
st.subheader("🔴 Топ-5 негативных слов по товару")
selected = st.selectbox("Выберите товар", sorted(df["product"].unique()))

prod_df = df[df["product"] == selected]
neg_words = top_negative_words(prod_df, n=5)
if neg_words:
    words, counts = zip(*neg_words)
    fig2, ax = plt.subplots(figsize=(6, 3))
    ax.barh(list(words), list(counts), color="coral", alpha=0.85)
    ax.set_title(f"Топ слов в негативных отзывах: {selected[:30]}")
    st.pyplot(fig2)
    plt.close()
else:
    st.info("Нет негативных отзывов по этому товару.")

# ── Word cloud per product ─────────────────────────────────────────────────────

st.subheader("☁️ Облако слов")
texts = " ".join(prod_df["text"].tolist())
if texts.strip():
    wc = WordCloud(width=800, height=300, background_color="white",
                   max_words=60, collocations=False).generate(texts)
    fig3, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig3)
    plt.close()

log(user_id, "visit_statistics")
