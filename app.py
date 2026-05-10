"""Main Streamlit page — review form, sentiment analysis, charts."""
import os
import uuid
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from wordcloud import WordCloud

from logger_setup import log

st.set_page_config(page_title="Анализ отзывов", page_icon="📊", layout="wide")

MODELS_DIR = "models_saved"


# ── User identity (session-persistent) ────────────────────────────────────────

def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


# ── First-run setup ────────────────────────────────────────────────────────────

def ensure_setup():
    from database import init_db, count_reviews, load_from_dataframe
    from dataset import generate_dataset, save_to_csv

    init_db()
    if count_reviews() == 0:
        with st.spinner("⏳ Генерация датасета (300 отзывов)…"):
            df = generate_dataset(300)
            save_to_csv(df)
            load_from_dataframe(df)
        st.success("Датасет создан!")

    if not os.path.exists(f"{MODELS_DIR}/vectorizer.pkl"):
        st.info("Первый запуск: обучение моделей (~5–10 мин). Пожалуйста, подождите…")
        with st.spinner("🤖 Обучение 5 моделей с GridSearchCV…"):
            from train_models import load_training_data, train_models, generate_report
            X, y, df = load_training_data()
            res, trained, vec, X_test, y_test = train_models(X, y)
            generate_report(df, res, trained, vec, y_test, X_test)
        st.success("Модели обучены! Перезапустите страницу.")
        st.stop()

    if not os.path.exists(f"{MODELS_DIR}/cluster_results.pkl"):
        with st.spinner("🔍 Вычисление эмбеддингов (SentenceTransformer)…"):
            from database import get_reviews
            from clustering import cluster_reviews
            cluster_reviews(get_reviews())


@st.cache_resource(show_spinner=False)
def load_models():
    if not os.path.exists(f"{MODELS_DIR}/vectorizer.pkl"):
        return None, None, None
    vec = joblib.load(f"{MODELS_DIR}/vectorizer.pkl")
    res = joblib.load(f"{MODELS_DIR}/results.pkl")
    models = {}
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "MLPClassifier", "CatBoost"]:
        p = f"{MODELS_DIR}/{name}.pkl"
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models, vec, res


@st.cache_data(ttl=30, show_spinner=False)
def load_data() -> pd.DataFrame:
    from database import get_reviews
    return get_reviews()


# ── Page ──────────────────────────────────────────────────────────────────────

ensure_setup()
user_id = get_user_id()
models, vectorizer, results = load_models()
df = load_data()

st.title("📊 Анализ отзывов клиентов")
st.caption(f"Сессия: пользователь #{user_id}")

# ── Review form ────────────────────────────────────────────────────────────────

st.header("✏️ Оставить отзыв")
c1, c2 = st.columns([3, 1])
with c1:
    review_text = st.text_area("Текст отзыва", placeholder="Напишите ваш отзыв…", height=100)
with c2:
    product = st.selectbox("Товар", sorted(df["product"].unique()))
    rating  = st.slider("Оценка", 1, 5, 4)

if st.button("Отправить отзыв", type="primary"):
    if not review_text.strip():
        st.warning("Введите текст отзыва")
    else:
        from database import add_review
        sentiment, predicted_label = None, "нейтральный"
        if models and vectorizer and rating != 3:
            from utils import predict_sentiment
            best = max(results, key=lambda n: results[n]["tuned"]["f1"])
            predicted_label, _ = predict_sentiment(review_text, models[best], vectorizer)
            sentiment = 1 if predicted_label == "позитивный" else 0
        add_review(review_text, rating, product, user_id, sentiment=sentiment)
        load_data.clear()
        log(user_id, "add_review", f"product={product} rating={rating} sentiment={predicted_label}")
        st.success(f"Отзыв добавлен! Предсказанная тональность: **{predicted_label}**")

# ── Sentiment analyser ─────────────────────────────────────────────────────────

st.header("🔍 Анализ тональности текста")
analyze_text = st.text_input("Введите любой текст для анализа тональности:")
if analyze_text.strip() and models and vectorizer:
    from utils import predict_sentiment
    best = max(results, key=lambda n: results[n]["tuned"]["f1"])
    label, conf = predict_sentiment(analyze_text, models[best], vectorizer)
    color = "green" if label == "позитивный" else "red"
    st.markdown(f"**Тональность:** :{color}[{label.upper()}]")
    st.progress(conf, text=f"Уверенность: {conf:.1%}")
    log(user_id, "predict_sentiment", f"label={label} conf={conf:.2f}")

# ── KPI cards ──────────────────────────────────────────────────────────────────

st.header("📈 Общая статистика")
df_b = df[df["sentiment"].notna()].copy()
df_b["sentiment"] = df_b["sentiment"].astype(float)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего отзывов",    len(df))
c2.metric("Средний рейтинг",  f"{df['rating'].mean():.2f} ⭐")
c3.metric("% позитивных",     f"{df_b['sentiment'].mean():.1%}" if len(df_b) > 0 else "—")
c4.metric("Уникальных товаров", df["product"].nunique())

# ── Charts ─────────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1:
    st.subheader("Распределение оценок")
    vc = df["rating"].value_counts().sort_index().reset_index()
    vc.columns = ["Оценка", "Количество"]
    st.plotly_chart(
        px.bar(vc, x="Оценка", y="Количество", color="Количество",
               color_continuous_scale="Blues", template="plotly_white"),
        use_container_width=True,
    )

with c2:
    st.subheader("Динамика отзывов по дням")
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    daily = df.groupby("date_only").size().reset_index(name="Отзывов")
    daily.columns = ["Дата", "Отзывов"]
    st.plotly_chart(
        px.line(daily, x="Дата", y="Отзывов", markers=True, template="plotly_white"),
        use_container_width=True,
    )

# ── Word clouds ────────────────────────────────────────────────────────────────

st.subheader("☁️ Облако слов")
c1, c2 = st.columns(2)
for col, sent_val, title, cmap in [
    (c1, 1, "Позитивные отзывы", "Greens"),
    (c2, 0, "Негативные отзывы", "Reds"),
]:
    with col:
        texts = " ".join(df[df["sentiment"] == sent_val]["text"].tolist())
        if texts.strip():
            wc = WordCloud(
                width=600, height=280, background_color="white",
                colormap=cmap, max_words=60, collocations=False,
            ).generate(texts)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            st.pyplot(fig)
            plt.close()
        else:
            st.info(f"Нет данных: {title}")

log(user_id, "visit_home")
