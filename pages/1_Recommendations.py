"""Страница персонализированных рекомендаций."""
import uuid
import streamlit as st
import pandas as pd
import plotly.express as px
from logger_setup import log

st.set_page_config(page_title="Рекомендации", layout="wide")
st.title("Персонализированные рекомендации")


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


@st.cache_data(ttl=30, show_spinner=False)
def load_data() -> pd.DataFrame:
    from database import get_reviews
    return get_reviews()


user_id = get_user_id()
st.caption(f"Пользователь: ID {user_id}")

df = load_data()

from recommender import get_recommendations, build_user_item_matrix

# ── User history ───────────────────────────────────────────────────────────────

matrix = build_user_item_matrix(df)
if user_id in matrix.index:
    history = matrix.loc[user_id].dropna().reset_index()
    history.columns = ["Товар", "Средняя оценка"]
    history["Средняя оценка"] = history["Средняя оценка"].round(1)
    if len(history) > 0:
        st.subheader("Ваши оценки")
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("Вы ещё не оставляли отзывов. Оставьте отзыв на главной странице!")
else:
    st.info("Вы новый пользователь — покажем популярные товары.")

# ── Recommendations ────────────────────────────────────────────────────────────

st.subheader("Рекомендуем вам")
with st.spinner("Вычисляю рекомендации…"):
    recs = get_recommendations(user_id, df, n=5)

if recs:
    for i, prod in enumerate(recs, 1):
        prod_df = df[df["product"] == prod]
        avg_rating = prod_df["rating"].mean()
        cnt = len(prod_df)
        b = prod_df[prod_df["sentiment"].notna()]
        pos = float(b["sentiment"].astype(float).mean()) if len(b) > 0 else 0.5

        c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
        c1.markdown(f"**{i}. {prod}**")
        c2.metric("Рейтинг", f"{avg_rating:.1f}")
        c3.metric("Отзывов", cnt)
        c4.metric("Позитив", f"{pos:.0%}")
        st.divider()
else:
    st.warning("Недостаточно данных для рекомендаций.")

# ── All products chart ─────────────────────────────────────────────────────────

st.subheader("Все товары по рейтингу")
prod_stats = df.groupby("product").agg(avg=("rating", "mean"), cnt=("rating", "count")).reset_index()
prod_stats.columns = ["Товар", "Средний рейтинг", "Отзывов"]
fig = px.scatter(
    prod_stats, x="Средний рейтинг", y="Отзывов", text="Товар",
    size="Отзывов", color="Средний рейтинг",
    color_continuous_scale="RdYlGn", template="plotly_white",
    title="Рейтинг vs количество отзывов",
)
fig.update_traces(textposition="top center")
st.plotly_chart(fig, use_container_width=True)

log(user_id, "visit_recommendations")
