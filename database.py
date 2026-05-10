import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "reviews.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            text       TEXT    NOT NULL,
            rating     INTEGER NOT NULL,
            sentiment  INTEGER,
            date       TEXT    NOT NULL,
            product    TEXT    NOT NULL,
            user_id    INTEGER NOT NULL,
            user_name  TEXT    NOT NULL,
            phone      TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def load_from_dataframe(df: pd.DataFrame):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        sentiment = None if pd.isna(row.get("sentiment")) else int(row["sentiment"])
        conn.execute(
            "INSERT INTO reviews (text, rating, sentiment, date, product, user_id, user_name, phone) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["text"], int(row["rating"]), sentiment,
                str(row["date"])[:10], row["product"],
                int(row["user_id"]), row["user_name"], row["phone"],
            ),
        )
    conn.commit()
    conn.close()


def add_review(text, rating, product, user_id, user_name="Гость", phone="+7900*******", sentiment=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO reviews (text, rating, sentiment, date, product, user_id, user_name, phone) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (text, int(rating), sentiment, datetime.now().strftime("%Y-%m-%d"), product, int(user_id), user_name, phone),
    )
    conn.commit()
    conn.close()


def get_reviews() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    return df


def count_reviews() -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.close()
    return n


def get_user_ratings(user_id: int) -> dict:
    df = get_reviews()
    user_df = df[df["user_id"] == user_id]
    return dict(zip(user_df["product"], user_df["rating"]))
