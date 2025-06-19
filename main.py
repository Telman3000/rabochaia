# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime
import os
import platform

from dotenv import load_dotenv
load_dotenv()

# Настройки подключения к MongoDB

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')  # URI подключения
DB_NAME   = os.getenv('DB_NAME', 'namaz_db')                      # имя БД

# Создаём клиент и выбираем БД
client    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db        = client[DB_NAME]

# Определяем коллекции
col_learners  = db['users_learners']   # коллекция с данными пользователей
col_logs      = db['users_logs']       # коллекция логов
col_outcomes  = db['outcomes']         # коллекция исходов (outcomes)
col_grouped   = db['users_grouped']    # коллекция агрегированных метрик (если есть)

# Инициализация FastAPI и шаблонизатора
app       = FastAPI(title="Dynamic Metrics Table")
templates = Jinja2Templates(directory="templates")


def console_log(action: str, message: str):
    """Печать логов с меткой времени для отладки."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{action} at {ts}] {message}")


# Загрузка данных из MongoDB

def load_learners_from_db() -> pd.DataFrame:
    """Загружает пользователей из users_learners в DataFrame."""
    docs = list(col_learners.find())
    df   = pd.DataFrame(docs)
    console_log("Debug", f"Learners columns: {df.columns.tolist()}")

    # Преобразуем _id в строку
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)

    # Оставляем только тех, у кого selected == 0
    if 'selected' in df.columns:
        df['selected'] = pd.to_numeric(df['selected'], errors='coerce') \
                           .fillna(0).astype(int)
        df = df[df['selected'] == 0]

    # Гарантируем наличие launch_count
    if 'launch_count' in df.columns:
        df['launch_count'] = pd.to_numeric(df['launch_count'], errors='coerce') \
                                 .fillna(0).astype(int)
    else:
        df['launch_count'] = 0

    # Проверка обязательного поля
    if 'recommendation_method' not in df.columns:
        raise KeyError("Нет поля 'recommendation_method' в коллекции users_learners")

    return df


def load_logs_from_db() -> pd.DataFrame:
    """Загружает логи из users_logs в DataFrame."""
    docs = list(col_logs.find())
    df   = pd.DataFrame(docs)
    if 'learner_id' in df.columns:
        df['learner_id'] = df['learner_id'].astype(str)
    return df


def build_outcome_map_db() -> dict:
    """
    Строит отображение Outcome ID -> список activity_id (Assesses).
    Берётся из коллекции outcomes.
    """
    outcome_map = defaultdict(list)
    for doc in col_outcomes.find():
        key = doc.get('Outcome ID') or doc.get('Outcome_ID') or doc.get('OutcomeID')
        assesses = doc.get('Assesses', '')
        for item in filter(None, map(str.strip, assesses.split(','))):
            outcome_map[key].append(item)
    return outcome_map


# ... остальные функции compute_grouped_size, compute_log_counts,
#     compute_ctr, compute_mastery, compute_retention, compute_engagement
#     остаются без изменений ...

# Роуты FastAPI

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    learners_df = load_learners_from_db()
    logs_df     = load_logs_from_db()
    outcome_map = build_outcome_map_db()

    # здесь объединяем и считаем метрики, передаём в шаблон
    # ...
    return templates.TemplateResponse("index.html", {
        "request": request,
        # передаём в шаблон готовый DataFrame .to_dict('records') и список колонок
    })


# Точка входа для Gunicorn (если вы запускаете через gunicorn main:app)
if __name__ != "__main__":
    import gunicorn.app.base

    class FastAPIApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, val in config.items():
                self.cfg.set(key.lower(), val)

        def load(self):
            return self.application

    port = int(os.getenv("PORT", 8000))
    console_log("Info", f"Starting Gunicorn on port {port}")

    FastAPIApplication(app, {
        "bind": f"0.0.0.0:{port}",
        "workers": 2,
        "accesslog": "-",
        "access_log_format":
            '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"',
    }).run()
