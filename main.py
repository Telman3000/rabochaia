# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime
import os

from dotenv import load_dotenv
load_dotenv()

# ─── Настройки MongoDB ────────────────────────────────────────────────────────────
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME   = os.getenv('DB_NAME', 'namaz_db')

client        = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db            = client[DB_NAME]
col_learners  = db['users_learners']
col_logs      = db['users_logs']
col_outcomes  = db['outcomes']
col_grouped   = db['users_grouped']

# ─── FastAPI + Jinja2 ─────────────────────────────────────────────────────────────
app       = FastAPI(title="Dynamic Metrics Table")
templates = Jinja2Templates(directory="templates")

def console_log(action: str, message: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{action} at {ts}] {message}")

# ─── Загрузка данных ──────────────────────────────────────────────────────────────
def load_learners_from_db() -> pd.DataFrame:
    docs = list(col_learners.find())
    df   = pd.DataFrame(docs)
    console_log("Debug", f"Learners columns: {df.columns.tolist()}")

    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)
    if 'selected' in df.columns:
        df['selected'] = pd.to_numeric(df['selected'], errors='coerce').fillna(0).astype(int)
        df = df[df['selected'] == 0]
    if 'launch_count' in df.columns:
        df['launch_count'] = pd.to_numeric(df['launch_count'], errors='coerce').fillna(0).astype(int)
    else:
        df['launch_count'] = 0

    if 'recommendation_method' not in df.columns:
        raise KeyError("Нет поля 'recommendation_method' в коллекции users_learners")
    return df

def load_logs_from_db() -> pd.DataFrame:
    docs = list(col_logs.find())
    df   = pd.DataFrame(docs)
    if 'learner_id' in df.columns:
        df['learner_id'] = df['learner_id'].astype(str)
    return df

def build_outcome_map_db() -> dict:
    outcome_map = defaultdict(list)
    for doc in col_outcomes.find():
        key      = doc.get('Outcome ID') or doc.get('Outcome_ID') or doc.get('OutcomeID')
        assesses = doc.get('Assesses', '')
        for item in filter(None, map(str.strip, assesses.split(','))):
            outcome_map[key].append(item)
    return outcome_map

# ─── Метрики ────────────────────────────────────────────────────────────────────
def compute_grouped_size(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('recommendation_method')\
             .agg(group_size=('_id','count'))\
             .reset_index()

def compute_retention(logs_df, learners_df):
    launch = logs_df[logs_df.get('activity_id') == 'launch']
    return (
        launch
        .groupby('learner_id').size().reset_index(name='launch_count')
        .merge(learners_df[['_id','recommendation_method']],
               left_on='learner_id', right_on='_id', how='left')
        .groupby('recommendation_method')['launch_count']
        .mean().reset_index(name='retention')
    )

def compute_engagement(logs_df, learners_df):
    return (
        logs_df
        .groupby('learner_id').size().reset_index(name='log_count')
        .merge(learners_df[['_id','recommendation_method']],
               left_on='learner_id', right_on='_id', how='left')
        .groupby('recommendation_method')['log_count']
        .mean().reset_index(name='engagement')
    )

def compute_ctr(logs_df, learners_df):
    ctr_logs = logs_df[logs_df.get('activity_id') == 'recommended_item_selected']
    return (
        ctr_logs
        .groupby('learner_id').size().reset_index(name='ctr_clicks')
        .merge(learners_df[['_id','recommendation_method']],
               left_on='learner_id', right_on='_id', how='left')
        .groupby('recommendation_method')['ctr_clicks']
        .sum().reset_index()
    )

def compute_mastery(logs_df, learners_df, outcome_map):
    if 'value' not in logs_df.columns:
        df = learners_df[['recommendation_method']].copy()
        df['mastery_rate'] = 0.0
        return df

    numeric = logs_df[logs_df['value'].apply(
        lambda x: isinstance(x, (int,float,str)) and str(x).replace('.','',1).isdigit()
    )].copy()
    numeric['score'] = numeric['value'].astype(float)

    item_score = numeric.groupby(['learner_id','activity_id'])['score']\
                        .max().unstack(fill_value=0)
    item_score['mastery_score'] = item_score.apply(
        lambda row: sum(
            1 for items in outcome_map.values()
            if any(row.get(i,0)>0 for i in items)
        ), axis=1
    )
    return (
        item_score[['mastery_score']].reset_index()
        .merge(learners_df[['_id','recommendation_method']],
               left_on='learner_id', right_on='_id', how='left')
        .groupby('recommendation_method')['mastery_score']
        .mean().reset_index(name='mastery_rate')
    )

# ─── Маршруты ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics", response_class=HTMLResponse)
async def metrics(request: Request):
    try:
        learners_df = load_learners_from_db()
        logs_df     = load_logs_from_db()
        outcome_map = build_outcome_map_db()

        grouped    = compute_grouped_size(learners_df)
        retention  = compute_retention(logs_df, learners_df)
        engagement = compute_engagement(logs_df, learners_df)
        ctr        = compute_ctr(logs_df, learners_df)
        mastery    = compute_mastery(logs_df, learners_df, outcome_map)

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
        html_table = table.to_html(classes="table table-striped", index=False)

        return templates.TemplateResponse("metrics.html", {
            "request": request,
            "table": html_table
        })
    except Exception as e:
        console_log("Error", f"Error in /metrics: {e}")
        return HTMLResponse(f"Ошибка: {e}", status_code=500)

# ─── Запуск ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    console_log("Info", f"Starting Uvicorn on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
