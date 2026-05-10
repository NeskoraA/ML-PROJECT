import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """User × product rating matrix (mean rating if multiple)."""
    ratings = df.groupby(["user_id", "product"])["rating"].mean().reset_index()
    return ratings.pivot(index="user_id", columns="product", values="rating")


def _filled(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.apply(lambda row: row.fillna(row.mean()), axis=1)


def get_similar_users(user_id: int, matrix: pd.DataFrame, n: int = 10) -> list[tuple[int, float]]:
    if user_id not in matrix.index:
        return []
    filled = _filled(matrix)
    user_vec = filled.loc[user_id].values.reshape(1, -1)
    sims = cosine_similarity(user_vec, filled.values)[0]
    pairs = [(uid, float(s)) for uid, s in zip(matrix.index, sims) if uid != user_id]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def _predict_rating(user_id: int, product: str, matrix: pd.DataFrame,
                    similar_users: list[tuple[int, float]]) -> float:
    ratings = []
    for sim_uid, sim in similar_users:
        if product in matrix.columns and sim_uid in matrix.index:
            r = matrix.loc[sim_uid, product]
            if not np.isnan(r):
                ratings.append((float(r), sim))
    if not ratings:
        col = matrix[product].dropna() if product in matrix.columns else pd.Series(dtype=float)
        return float(col.mean()) if len(col) > 0 else 3.0
    num = sum(r * s for r, s in ratings)
    den = sum(s for _, s in ratings)
    return num / den if den > 0 else 3.0


def get_recommendations(user_id: int, df: pd.DataFrame, n: int = 5) -> list[str]:
    matrix = build_user_item_matrix(df)

    if user_id not in matrix.index:
        avg = df.groupby("product")["rating"].mean().sort_values(ascending=False)
        return list(avg.head(n).index)

    similar = get_similar_users(user_id, matrix)
    rated = set(matrix.loc[user_id].dropna().index)
    unrated = set(matrix.columns) - rated

    if not unrated:
        return list(matrix.loc[user_id].sort_values(ascending=False).head(n).index)

    preds = [(p, _predict_rating(user_id, p, matrix, similar)) for p in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in preds[:n]]
