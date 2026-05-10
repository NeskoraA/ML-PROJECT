"""
Pytest unit-tests (5+).
Run from project root:  pytest tests/
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── 1. mask_phone ──────────────────────────────────────────────────────────────

def test_mask_phone_hides_middle():
    from utils import mask_phone
    result = mask_phone("+79001234567")
    assert result.startswith("+790")
    assert result.endswith("4567")
    assert "***" in result


def test_mask_phone_shorter_than_output():
    from utils import mask_phone
    original = "+79001234567"
    assert len(mask_phone(original)) < len(original)


def test_mask_phone_short_input():
    from utils import mask_phone
    # Should not crash on short string
    result = mask_phone("123")
    assert isinstance(result, str)


# ── 2. mask_name ───────────────────────────────────────────────────────────────

def test_mask_name_keeps_first_two():
    from utils import mask_name
    result = mask_name("Иванов Алексей")
    assert result[:2] == "Ив"


def test_mask_name_contains_asterisks():
    from utils import mask_name
    result = mask_name("Иванов Алексей")
    assert "*" in result


def test_mask_name_very_short():
    from utils import mask_name
    result = mask_name("Ли")
    assert result.startswith("Л")
    assert len(result) == 2


# ── 3. predict_sentiment ───────────────────────────────────────────────────────

def _make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts  = ["отлично", "замечательно", "прекрасно", "хорошо", "рекомендую",
              "ужасно",  "плохо",        "брак",      "разочарован", "кошмар"]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    vec = TfidfVectorizer()
    X   = vec.fit_transform(texts)
    m   = LogisticRegression().fit(X, labels)
    return m, vec


def test_predict_sentiment_returns_tuple():
    from utils import predict_sentiment
    m, vec = _make_model()
    result = predict_sentiment("отличный товар", m, vec)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_predict_sentiment_label_valid():
    from utils import predict_sentiment
    m, vec = _make_model()
    label, conf = predict_sentiment("ужасное качество", m, vec)
    assert label in ("позитивный", "негативный")


def test_predict_sentiment_confidence_range():
    from utils import predict_sentiment
    m, vec = _make_model()
    _, conf = predict_sentiment("замечательно", m, vec)
    assert 0.0 <= conf <= 1.0


# ── 4. Dataset loading ─────────────────────────────────────────────────────────

def test_generate_dataset_length():
    from dataset import generate_dataset
    df = generate_dataset(50)
    assert len(df) == 50


def test_generate_dataset_columns():
    from dataset import generate_dataset
    df = generate_dataset(50)
    for col in ("text", "rating", "product", "user_id", "user_name", "phone", "date"):
        assert col in df.columns, f"Missing column: {col}"


def test_generate_dataset_ratings_valid():
    from dataset import generate_dataset
    df = generate_dataset(100)
    assert df["rating"].between(1, 5).all()


def test_generate_dataset_reproducible():
    from dataset import generate_dataset
    df1 = generate_dataset(30, seed=0)
    df2 = generate_dataset(30, seed=0)
    assert list(df1["product"]) == list(df2["product"])


# ── 5. compute_metrics ────────────────────────────────────────────────────────

def test_compute_metrics_accuracy():
    from train_models import compute_metrics
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.8)


def test_compute_metrics_all_keys():
    from train_models import compute_metrics
    m = compute_metrics(np.array([1, 0]), np.array([1, 0]))
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in m


def test_compute_metrics_perfect():
    from train_models import compute_metrics
    y = np.array([1, 0, 1, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"]       == pytest.approx(1.0)


# ── 6. Recommendations ───────────────────────────────────────────────────────

def test_recommendations_returns_list():
    from dataset import generate_dataset
    from recommender import get_recommendations
    df = generate_dataset(100)
    result = get_recommendations(1, df, n=3)
    assert isinstance(result, list)


def test_recommendations_max_n():
    from dataset import generate_dataset
    from recommender import get_recommendations
    df = generate_dataset(100)
    result = get_recommendations(1, df, n=3)
    assert len(result) <= 3


def test_recommendations_unknown_user():
    from dataset import generate_dataset
    from recommender import get_recommendations
    df = generate_dataset(100)
    result = get_recommendations(9999, df, n=5)
    # Should fall back to popular products, not crash
    assert isinstance(result, list)


# ── 7. top_positive_products ────────────────────────────────────────────────

def test_top_positive_products_columns():
    from dataset import generate_dataset
    from utils import top_positive_products
    df = generate_dataset(100)
    top = top_positive_products(df, n=3)
    assert "positive_ratio" in top.columns
    assert "product" in top.columns


def test_top_positive_products_max_n():
    from dataset import generate_dataset
    from utils import top_positive_products
    df = generate_dataset(100)
    assert len(top_positive_products(df, n=3)) <= 3


# ── 8. top_negative_words ────────────────────────────────────────────────────

def test_top_negative_words_list():
    from dataset import generate_dataset
    from utils import top_negative_words
    df = generate_dataset(100)
    result = top_negative_words(df, n=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_top_negative_words_tuples():
    from dataset import generate_dataset
    from utils import top_negative_words
    df = generate_dataset(100)
    result = top_negative_words(df, n=5)
    for item in result:
        assert len(item) == 2
        assert isinstance(item[1], int)
