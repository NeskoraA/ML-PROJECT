"""
Run once to initialise data, train all 5 models with GridSearchCV,
and generate student_report.png.

    python train_models.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import save_npz, load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

MODELS_DIR = "models_saved"


# ── Data setup ────────────────────────────────────────────────────────────────

def setup_data():
    from database import init_db, count_reviews, load_from_dataframe
    from dataset import generate_dataset, save_to_csv
    init_db()
    if count_reviews() == 0:
        print("Generating dataset (300 reviews)…")
        df = generate_dataset(300)
        save_to_csv(df)
        load_from_dataframe(df)
        print(f"  {len(df)} records loaded into reviews.db")
    else:
        from database import count_reviews as cr
        print(f"Database already has {cr()} reviews")


def load_training_data():
    from database import get_reviews
    df = get_reviews()
    df_b = df[df["sentiment"].notna()].copy()
    df_b["sentiment"] = df_b["sentiment"].astype(int)
    return df_b["text"].values, df_b["sentiment"].values, df


# ── Training ──────────────────────────────────────────────────────────────────

BASE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "RandomForest":       RandomForestClassifier(random_state=42),
    "XGBoost":            XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42),
    "MLPClassifier":      MLPClassifier(max_iter=500, random_state=42),
    "CatBoost":           CatBoostClassifier(verbose=False, random_seed=42),
}

PARAM_GRIDS = {
    "LogisticRegression": {
        "C":         [0.1, 1.0, 10.0],
        "solver":    ["lbfgs", "saga"],
        "penalty":   ["l2"],
    },
    "RandomForest": {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.3],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
        "alpha":              [0.0001, 0.001, 0.01],
        "activation":         ["relu", "tanh"],
    },
    "CatBoost": {
        "iterations":    [100, 200, 300],
        "depth":         [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def train_models(X, y):
    os.makedirs(MODELS_DIR, exist_ok=True)

    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y,
    )

    results = {}
    trained = {}

    for name, base in BASE_MODELS.items():
        print(f"\n[{name}]")

        # Base (no tuning)
        base.fit(X_train, y_train)
        base_metrics = compute_metrics(y_test, base.predict(X_test))
        print(f"  Base  F1={base_metrics['f1']:.3f}  Acc={base_metrics['accuracy']:.3f}")

        # GridSearchCV
        n_jobs = 1 if name == "CatBoost" else -1
        gs = GridSearchCV(
            BASE_MODELS[name].__class__(**{
                k: v for k, v in BASE_MODELS[name].get_params().items()
            }),
            PARAM_GRIDS[name],
            cv=3, scoring="f1", n_jobs=n_jobs, refit=True,
        )
        # CatBoost: rebuild cleanly so verbose=False is always set
        if name == "CatBoost":
            gs = GridSearchCV(
                CatBoostClassifier(verbose=False, random_seed=42),
                PARAM_GRIDS[name], cv=3, scoring="f1", n_jobs=1, refit=True,
            )

        gs.fit(X_train, y_train)
        tuned_metrics = compute_metrics(y_test, gs.predict(X_test))
        print(f"  Tuned F1={tuned_metrics['f1']:.3f}  Acc={tuned_metrics['accuracy']:.3f}  best={gs.best_params_}")

        results[name] = {
            "base":        base_metrics,
            "tuned":       tuned_metrics,
            "best_params": gs.best_params_,
        }
        trained[name] = gs.best_estimator_
        joblib.dump(gs.best_estimator_, f"{MODELS_DIR}/{name}.pkl")

    joblib.dump(vectorizer, f"{MODELS_DIR}/vectorizer.pkl")
    joblib.dump(results,    f"{MODELS_DIR}/results.pkl")
    save_npz(f"{MODELS_DIR}/X_test.npz", X_test)
    np.save(f"{MODELS_DIR}/y_test.npy", y_test)

    return results, trained, vectorizer, X_test, y_test


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(df, results, trained, vectorizer, y_test, X_test):
    from utils import top_positive_products, top_negative_words, predict_sentiment

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Анализ отзывов — Итоговый отчёт студента", fontsize=15, fontweight="bold")

    df_b = df[df["sentiment"].notna()].copy()
    df_b["sentiment"] = df_b["sentiment"].astype(float)

    # 1. Rating distribution
    ax = axes[0, 0]
    vc = df["rating"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="royalblue", alpha=0.8, edgecolor="white")
    ax.set_title("Распределение оценок")
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Количество")

    # 2. Sentiment pie
    ax = axes[0, 1]
    sc = df_b["sentiment"].value_counts()
    labels_pie = ["Позитив" if i == 1 else "Негатив" for i in sc.index]
    ax.pie(sc.values, labels=labels_pie, colors=["lightgreen", "coral"], autopct="%1.1f%%")
    ax.set_title("Соотношение тональности")

    # 3. Top products by positive ratio
    ax = axes[0, 2]
    tp = top_positive_products(df, n=5)
    ax.barh(tp["product"].str[:22], tp["positive_ratio"] * 100, color="seagreen", alpha=0.8)
    ax.set_title("Топ-5 товаров по % позитива")
    ax.set_xlabel("% позитивных отзывов")

    # 4. Daily dynamics
    ax = axes[1, 0]
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    daily = df.groupby("date_only").size().reset_index(name="cnt")
    ax.plot(range(len(daily)), daily["cnt"], "b-o", linewidth=2, markersize=3)
    ax.set_title("Динамика отзывов по дням")
    ax.set_ylabel("Количество")

    # 5. Top negative words
    ax = axes[1, 1]
    nw = top_negative_words(df, n=10)
    if nw:
        words, counts = zip(*nw)
        ax.barh(list(words), list(counts), color="coral", alpha=0.8)
    ax.set_title("Топ слов в негативных отзывах")

    # 6. F1 before/after GridSearchCV
    ax = axes[1, 2]
    if results:
        names = list(results.keys())
        b_f1 = [results[n]["base"]["f1"] for n in names]
        t_f1 = [results[n]["tuned"]["f1"] for n in names]
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], b_f1, 0.35, label="До", color="lightblue")
        ax.bar([i + 0.2 for i in x], t_f1, 0.35, label="После", color="royalblue")
        ax.set_xticks(x)
        ax.set_xticklabels([n[:8] for n in names], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("F1-score: до и после GridSearchCV")
        ax.legend()

    # 7. Product review count
    ax = axes[2, 0]
    pc = df["product"].value_counts()
    ax.bar(range(len(pc)), pc.values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(pc)))
    ax.set_xticklabels([p[:12] for p in pc.index], rotation=40, ha="right")
    ax.set_title("Отзывов по товарам")

    # 8. Example predictions
    ax = axes[2, 1]
    if results and trained:
        best_name = max(results.items(), key=lambda x: x[1]["tuned"]["f1"])[0]
        phrases = [
            "Отличный товар, всем рекомендую!",
            "Ужасное качество, разочарован",
            "Нормальный товар, ничего особенного",
        ]
        confs, colors_bar = [], []
        for ph in phrases:
            lbl, conf = predict_sentiment(ph, trained[best_name], vectorizer)
            confs.append(conf)
            colors_bar.append("lightgreen" if lbl == "позитивный" else "coral")
        ax.barh([p[:30] for p in phrases], confs, color=colors_bar, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_title(f"Примеры предсказаний ({best_name})")

    # 9. Metrics summary table
    ax = axes[2, 2]
    ax.axis("off")
    if results:
        tdata = [
            [n, f"{r['tuned']['accuracy']:.3f}", f"{r['tuned']['precision']:.3f}",
             f"{r['tuned']['recall']:.3f}", f"{r['tuned']['f1']:.3f}"]
            for n, r in results.items()
        ]
        tbl = ax.table(
            cellText=tdata,
            colLabels=["Модель", "Acc", "Prec", "Rec", "F1"],
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 2.2)
        ax.set_title("Итоговые метрики (после GridSearchCV)")

    plt.tight_layout()
    plt.savefig("student_report.png", dpi=150, bbox_inches="tight")
    print("\nSaved: student_report.png")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_data()
    X, y, df = load_training_data()
    print(f"\nTraining data: {len(X)} samples  pos={y.sum()}  neg={(1-y).sum()}")

    results, trained, vectorizer, X_test, y_test = train_models(X, y)

    print("\n" + "=" * 80)
    print(f"{'Модель':<22} {'Acc (до)':<10} {'Acc (посл)':<12} {'F1 (до)':<10} {'F1 (посл)':<10}  Параметры")
    print("-" * 80)
    for n, r in results.items():
        print(f"{n:<22} {r['base']['accuracy']:<8.3f} {r['tuned']['accuracy']:<8.3f} "
              f"{r['base']['f1']:<8.3f} {r['tuned']['f1']:<8.3f}  {r['best_params']}")

    generate_report(df, results, trained, vectorizer, y_test, X_test)
    print("\nDone! Run:  streamlit run app.py")
