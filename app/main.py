from pathlib import Path
import json
from . import (config, ml, models, db, schema)
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from cassandra.cqlengine.management import sync_table
from cassandra.query import SimpleStatement

app = FastAPI()
settings = config.get_settings()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-metadata.json"

AI_MODEL = None
DB_SESSION = None
SPAMInference = models.SPAMInference

@app.on_event("startup")
def on_start_up():
    global AI_MODEL, DB_SESSION
    AI_MODEL = ml.MLModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )
    DB_SESSION = db.get_session()
    sync_table(SPAMInference)


@app.get("/")  # /?q is a url parameter
def home():
    return {'home': "SPAM ML CLASSIFIER"}

@app.post("/")  # /?q is a url parameter
def create_inference(query:schema.Query):
    global AI_MODEL
    preds_dict = AI_MODEL.predict_text(query.q)
    top = preds_dict.get('top prediction')
    data = {'query': query.q, **top}
    obj = SPAMInference.objects.create(**data)
    return obj
    # return {'query': query, 'results': preds_dict, 'db_client_id':settings.db_client_id}


@app.get("/inferences/{my_uuid}")
def read_inference(my_uuid):
    obj = SPAMInference.objects.get(uuid=my_uuid)
    return obj


@app.get("/inferences")
def list_inferences():
    q = SPAMInference.objects.all()
    print(q)
    return list(q)


@app.get("/dataset")
def export_inferences_():
    cql_query = "SELECT * FROM spam_inferences.spaminference LIMIT 10000"
    rows = DB_SESSION.execute(cql_query)
    return list(rows)


def fetch_rows(stmt:SimpleStatement, fetch_size:int=25, session=None):
    stmt.fetch_size = fetch_size
    result_set = session.execute(stmt)
    has_pages = result_set.has_more_pages
    yield "uuid, label, confidence_score, query \n"
    for row in result_set.current_rows:
        yield f"{row['uuid']}, {row['label']}, {row['confidence_score']}, {row['query']} \n"
    result_set = session.execute(stmt, paging_state=result_set.paging_state)
    while has_pages:
        for row in result_set.current_rows:
            yield f"{row['uuid']}, {row['label']}, {row['confidence_score']}, {row['query']} \n"
        has_pages = result_set.has_more_pages
        result_set = session.execute(stmt, paging_state=result_set.paging_state)


@app.get("/show-data")
def export_inferences():
    global DB_SESSION
    cql_query = "SELECT * FROM spam_inferences.spaminference LIMIT 10000"
    statement = SimpleStatement(cql_query)
    return StreamingResponse(fetch_rows(statement, 25, DB_SESSION))
