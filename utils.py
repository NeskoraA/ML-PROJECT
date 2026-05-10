import re
from collections import Counter


def mask_phone(phone: str) -> str:
    """Mask phone: +79001234567 → +790***4567"""
    phone = str(phone)
    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone


def mask_name(name: str) -> str:
    """Mask name keeping first 2 and last 2 visible chars."""
    name = str(name)
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * max(len(name) - 4, 1) + name[-2:]


def predict_sentiment(text: str, model, vectorizer) -> tuple[str, float]:
    """Return (label, confidence). Label is 'позитивный' or 'негативный'."""
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])
    label = "позитивный" if pred == 1 else "негативный"
    return label, confidence


_STOPWORDS = {
    "это", "как", "так", "все", "очень", "уже", "его", "что", "для", "при",
    "ещё", "ещe", "ни", "но", "или", "что", "себе", "нет", "есть", "был",
    "бы", "же", "мне", "мы", "вы", "он", "она", "они", "тут", "там",
    "здесь", "чем", "кто", "этот", "эта", "эти", "тот", "та", "те", "сам",
    "раз", "два", "три", "теперь", "потом", "когда", "после", "чтобы",
    "если", "только", "даже", "хотя", "тоже", "также", "зато", "сразу",
    "буду", "быть", "более", "можно", "надо", "нужно", "пока", "свой",
    "своя", "своё", "свои", "такой", "такая", "такие", "само",
}


def top_positive_products(df, n: int = 5):
    """DataFrame with top N products sorted by positive review ratio."""
    df_b = df[df["sentiment"].notna()].copy()
    df_b["sentiment"] = df_b["sentiment"].astype(float)
    stats = df_b.groupby("product").agg(
        total=("sentiment", "count"),
        positive=("sentiment", "sum"),
    ).reset_index()
    stats["positive_ratio"] = stats["positive"] / stats["total"]
    return stats.nlargest(n, "positive_ratio")[["product", "total", "positive", "positive_ratio"]]


def top_negative_words(df, n: int = 5) -> list[tuple[str, int]]:
    """Top N words (by frequency) found in negative reviews."""
    neg_texts = df[df["sentiment"] == 0]["text"].str.lower()
    words = []
    for text in neg_texts:
        words.extend(re.findall(r"[а-яёa-z]{4,}", text))
    words = [w for w in words if w not in _STOPWORDS]
    return Counter(words).most_common(n)
