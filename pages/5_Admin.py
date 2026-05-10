"""Страница администратора — последние 50 записей из app.log."""
import uuid
import streamlit as st
import pandas as pd
from logger_setup import read_last_logs, log

st.set_page_config(page_title="Администратор", page_icon="🔐", layout="wide")
st.title("🔐 Панель администратора")
st.caption("Только для разработчика / администратора")


def get_user_id() -> int:
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(uuid.uuid4().int % 30) + 1
    return st.session_state.user_id


user_id = get_user_id()

# ── Auth (simple password) ─────────────────────────────────────────────────────

ADMIN_PASSWORD = "admin123"

if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

if not st.session_state.admin_auth:
    pwd = st.text_input("Пароль администратора", type="password")
    if st.button("Войти"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# ── Log viewer ─────────────────────────────────────────────────────────────────

st.success("Вы вошли как администратор")
if st.button("🔄 Обновить"):
    st.rerun()

lines = read_last_logs(50)
st.subheader(f"📋 Последние {len(lines)} записей из app.log")

if not lines:
    st.info("Лог-файл пуст или не создан.")
else:
    rows = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3:
            rows.append({
                "Время":       parts[0].strip(),
                "user_id":     parts[1].strip(),
                "Действие":    parts[2].strip(),
                "Результат":   parts[3].strip() if len(parts) > 3 else "",
            })
        else:
            rows.append({"Время": line, "user_id": "", "Действие": "", "Результат": ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.download_button("📥 Скачать лог", "\n".join(lines), file_name="app.log", mime="text/plain")

# ── DB stats ───────────────────────────────────────────────────────────────────

st.subheader("🗄️ Состояние базы данных")
from database import get_reviews
df = get_reviews()
c1, c2, c3 = st.columns(3)
c1.metric("Записей в БД",   len(df))
c2.metric("Уникальных пользователей", df["user_id"].nunique())
c3.metric("Уникальных товаров", df["product"].nunique())

log(user_id, "visit_admin")
