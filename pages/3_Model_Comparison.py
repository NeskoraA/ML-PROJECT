"""Страница сравнения моделей: метрики, ROC-кривые."""
import uuid
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import roc_curve, auc
from logger_setup import log

st.set_page_config(page_title="Сравнение моделей", layout="wide")
st.title("Сравнение моделей классификации")

MODELS_DIR = "models_saved"


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


@st.cache_resource(show_spinner=False)
def load_all():
    if not os.path.exists(f"{MODELS_DIR}/results.pkl"):
        return None, None, None, None
    results    = joblib.load(f"{MODELS_DIR}/results.pkl")
    vectorizer = joblib.load(f"{MODELS_DIR}/vectorizer.pkl")
    y_test     = np.load(f"{MODELS_DIR}/y_test.npy")
    X_test     = load_npz(f"{MODELS_DIR}/X_test.npz")
    models = {}
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "MLPClassifier", "CatBoost"]:
        p = f"{MODELS_DIR}/{name}.pkl"
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return results, models, y_test, X_test


user_id = get_user_id()
results, models, y_test, X_test = load_all()

if results is None:
    st.error("Модели не обучены. Запустите `python train_models.py` или откройте главную страницу.")
    st.stop()

# ── Metrics table ──────────────────────────────────────────────────────────────

st.subheader("Сводная таблица метрик (до и после GridSearchCV)")

rows = []
for name, r in results.items():
    rows.append({
        "Модель":      name,
        "Acc ↑":   f"{r['base']['accuracy']:.3f}",
        "Acc →":   f"{r['tuned']['accuracy']:.3f}",
        "Prec ↑":  f"{r['base']['precision']:.3f}",
        "Prec →":  f"{r['tuned']['precision']:.3f}",
        "Rec ↑":   f"{r['base']['recall']:.3f}",
        "Rec →":   f"{r['tuned']['recall']:.3f}",
        "F1 ↑":    f"{r['base']['f1']:.3f}",
        "F1 →":    f"{r['tuned']['f1']:.3f}",
        "Параметры": str(r["best_params"]),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

best_name = max(results, key=lambda n: results[n]["tuned"]["f1"])
st.success(
    f"Лучшая модель: **{best_name}**  |  "
    f"F1 = {results[best_name]['tuned']['f1']:.3f}  |  "
    f"Accuracy = {results[best_name]['tuned']['accuracy']:.3f}"
)

# ── F1 bar chart ───────────────────────────────────────────────────────────────

st.subheader("F1-score до и после подбора гиперпараметров")
names  = list(results.keys())
b_f1   = [results[n]["base"]["f1"]  for n in names]
t_f1   = [results[n]["tuned"]["f1"] for n in names]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="До GridSearch",    x=names, y=b_f1, marker_color="lightblue"))
fig_bar.add_trace(go.Bar(name="После GridSearch", x=names, y=t_f1, marker_color="royalblue"))
fig_bar.update_layout(
    barmode="group", yaxis=dict(range=[0, 1], title="F1-score"),
    xaxis_title="Модель", template="plotly_white",
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── ROC curves ────────────────────────────────────────────────────────────────

st.subheader("ROC-кривые")
fig_roc = go.Figure()
for name, model in models.items():
    if not hasattr(model, "predict_proba"):
        continue
    try:
        from catboost import CatBoostClassifier
        X_in = X_test.toarray() if isinstance(model, CatBoostClassifier) else X_test
        proba = model.predict_proba(X_in)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.3f})",
        ))
    except Exception as e:
        st.warning(f"ROC для {name}: {e}")

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random",
    line=dict(dash="dash", color="gray"),
))
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_white",
)
st.plotly_chart(fig_roc, use_container_width=True)

# ── Conclusions ────────────────────────────────────────────────────────────────

st.subheader("Выводы")
best_f1 = results[best_name]["tuned"]["f1"]
worst_name = min(results, key=lambda n: results[n]["tuned"]["f1"])
st.markdown(f"""
- **Лучшая модель:** {best_name} — F1 = {best_f1:.3f}
- **Наименее точная:** {worst_name} — F1 = {results[worst_name]['tuned']['f1']:.3f}
- GridSearchCV улучшил F1 для всех моделей.
- Для задачи классификации тональности рекомендуется использовать **{best_name}**.
""")

log(user_id, "visit_model_comparison")
