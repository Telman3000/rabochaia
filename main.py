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


# Настройки подключения к MongoDB

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')  # URI подключения
DB_NAME   = 'namaz_db'                                           # имя БД

# Создаём клиент и выбираем БД
client    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db        = client[DB_NAME]

# Определяем коллекции
col_learners = db['users_learners']  # коллекция с данными пользователей
col_logs     = db['users_logs']      # коллекция логов
col_outcomes = db['outcomes']        # коллекция исходов (outcomes)
col_raw      = db['users_raw']       # коллекция для «сырых» данных
col_grouped  = db['users_grouped']   # коллекция агрегированных метрик


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

    # Оставляем только тех, у кого selected == 0 (если есть такое поле)
    if 'selected' in df.columns:
        df['selected'] = pd.to_numeric(df['selected'], errors='coerce')\
                           .fillna(0).astype(int)
        df = df[df['selected'] == 0]

    # Гарантируем наличие launch_count
    if 'launch_count' in df.columns:
        df['launch_count'] = pd.to_numeric(df['launch_count'], errors='coerce')\
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
        # несколько возможных ключей
        key = doc.get('Outcome ID') or doc.get('Outcome_ID') or doc.get('OutcomeID')
        assesses = doc.get('Assesses', '')
        for item in filter(None, map(str.strip, assesses.split(','))):
            outcome_map[key].append(item)
    return outcome_map


# Вычисление метрик

def compute_grouped_size(df: pd.DataFrame) -> pd.DataFrame:
    """Размер групп по recommendation_method."""
    return df.groupby('recommendation_method')\
             .agg(group_size=('_id', 'count'))\
             .reset_index()

def compute_retention(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    """Среднее число запусков (activity_id=='launch') по каждому методу."""
    launch = logs_df[logs_df.get('activity_id') == 'launch']
    return (
        launch
        .groupby('learner_id')
        .size()
        .reset_index(name='launch_count')
        # ИСПРАВЛЕНО: убран пробел перед '_id'
        .merge(
            learners_df[['_id', 'recommendation_method']],
            left_on='learner_id', right_on='_id', how='left'
        )
        .groupby('recommendation_method')['launch_count']
        .mean()
        .reset_index(name='retention')
    )

def compute_engagement(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    """Среднее число всех логов по пользователю для каждого метода."""
    return (
        logs_df
        .groupby('learner_id')
        .size()
        .reset_index(name='log_count')
        .merge(
            learners_df[['_id', 'recommendation_method']],
            left_on='learner_id', right_on='_id', how='left'
        )
        .groupby('recommendation_method')['log_count']
        .mean()
        .reset_index(name='engagement')
    )

def compute_ctr(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    """CTR: суммарное число кликов по recommended_item_selected на метод."""
    ctr_logs = logs_df[logs_df.get('activity_id') == 'recommended_item_selected']
    return (
        ctr_logs
        .groupby('learner_id')
        .size()
        .reset_index(name='ctr_clicks')
        .merge(
            learners_df[['_id', 'recommendation_method']],
            left_on='learner_id', right_on='_id', how='left'
        )
        .groupby('recommendation_method')['ctr_clicks']
        .sum()
        .reset_index()
    )

def compute_mastery(logs_df: pd.DataFrame, learners_df: pd.DataFrame, outcome_map: dict) -> pd.DataFrame:
    """Mastery rate: среднее число освоенных Outcome ID по методу."""
    # Если нет поля value — сразу нули
    if 'value' not in logs_df.columns:
        df = learners_df[['recommendation_method']].copy()
        df['mastery_rate'] = 0.0
        return df

    # Оставляем только записи с числовым score
    numeric = logs_df[logs_df['value'].apply(
        lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()
    )].copy()
    numeric['score'] = numeric['value'].astype(float)

    # Максимальный score по комбинации learner_id + activity_id
    item_score = numeric.groupby(['learner_id', 'activity_id'])['score']\
                        .max().unstack(fill_value=0)
    # Считаем mastery_score (сколько Outcome освоено)
    item_score['mastery_score'] = item_score.apply(
        lambda row: sum(
            1 for items in outcome_map.values()
            if any(row.get(i, 0) > 0 for i in items)
        ), axis=1
    )
    return (
        item_score[['mastery_score']]
        .reset_index()
        .merge(
            learners_df[['_id', 'recommendation_method']],
            left_on='learner_id', right_on='_id', how='left'
        )
        .groupby('recommendation_method')['mastery_score']
        .mean()
        .reset_index(name='mastery_rate')
    )


# HTTP-маршруты

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Главная страница — просто кнопка 'Рассчитать метрики'."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics", response_class=HTMLResponse)
def metrics(request: Request):
    """Запускает загрузку, вычисляет метрики и отдает HTML-таблицу."""
    try:
        # 1) Загружаем данные
        learners_df = load_learners_from_db()
        logs_df     = load_logs_from_db()
        outcome_map = build_outcome_map_db()

        # 2) Сохраняем «сырые» данные в отдельную коллекцию
        raw_docs = learners_df.to_dict(orient="records")
        col_raw.delete_many({})
        if raw_docs:
            col_raw.insert_many(raw_docs)

        # 3) Считаем все метрики
        grouped    = compute_grouped_size(learners_df)
        retention  = compute_retention(logs_df, learners_df)
        engagement = compute_engagement(logs_df, learners_df)
        ctr        = compute_ctr(logs_df, learners_df)
        mastery    = compute_mastery(logs_df, learners_df, outcome_map)

        # 4) Объединяем, считаем CTR и готовим финальную таблицу
        final = (
            grouped
            .merge(retention,  on="recommendation_method", how="left")
            .merge(engagement, on="recommendation_method", how="left")
            .merge(ctr,        on="recommendation_method", how="left")
            .merge(mastery,    on="recommendation_method", how="left")
            .fillna(0)
        )
        final["CTR"] = final["ctr_clicks"] / final["group_size"]

        table = final[[
            "recommendation_method", "group_size",
            "retention", "engagement", "CTR", "mastery_rate"
        ]]

        # 5) Сохраняем агрегаты в MongoDB
        grouped_docs = table.to_dict(orient="records")
        col_grouped.delete_many({})
        if grouped_docs:
            col_grouped.insert_many(grouped_docs)

        # 6) Рендерим HTML через pandas
        html_table = table.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("metrics.html", {
            "request": request,
            "table": html_table
        })

    except Exception as e:
        console_log("Error", f"Error in metrics route: {e}")
        return HTMLResponse(f"Ошибка: {e}", status_code=500)


# Запуск приложения

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    if platform.system() == "Windows":
        import uvicorn
        console_log("Info", f"Starting uvicorn on port {port}")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    else:
        from gunicorn.app.base import Application
        from gunicorn.glogging import Logger
        import fcntl

        class CustomAccessLogger(Logger):
            """Исключаем статику и системные запросы."""
            def access(self, resp, req, environ, request_time):
                ua   = environ.get("HTTP_USER_AGENT", "")
                path = environ.get("PATH_INFO", "")
                if path.startswith("/static/") or "Go-http-client" in ua:
                    return
                super().access(resp, req, environ, request_time)

        class FastAPIApplication(Application):
            def init(self, parser, opts, args):
                return {
                    "bind":            f"0.0.0.0:{port}",
                    "workers":         2,
                    "accesslog":       "-",
                    "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"',
                    "logger_class":    CustomAccessLogger
                }
            def load(self):
                return app

        console_log("Info", f"Starting Gunicorn on port {port}")
        FastAPIApplication().run()


# Формат логов Gunicorn:
# %(h)s – IP-адрес клиента (remote address)
# %(l)s – идентификатор клиента (обычно всегда "-", т.к. identd редко используется)
# %(u)s – имя пользователя при HTTP-авторизации (обычно "-" если авторизации нет)
# %(t)s – время обработки запроса в формате [дд/МММ/гггг:чч:мм:сс +зона]
# %(r)s – первая строка HTTP-запроса (метод, путь, версия), например "GET /metrics HTTP/1.1"
# %(s)s – HTTP-статус ответа (200, 404, 500 и т.п.)
# %(b)s – размер тела ответа в байтах (без заголовков)
# %(f)s – значение заголовка Referer (откуда пришёл запрос)
# %(a)s – значение заголовка User-Agent (браузер или клиент)