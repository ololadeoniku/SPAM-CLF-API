from pathlib import Path
import json
from . import (config, ml)
from typing import Optional
from fastapi import FastAPI


app = FastAPI()
settings = config.get_settings()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-metadata.json"

AI_MODEL = None

@app.on_event("startup")
def on_start_up():
    global AI_MODEL
    AI_MODEL = ml.MLModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )


@app.get("/")  # /?q is a url parameter
def read_index(q:Optional[str]=None):
    global AI_MODEL
    query = q or "hello from spam classifier"
    preds_dict = AI_MODEL.predict_text(query)
    return {'query': query, 'results': preds_dict, 'db_client_id':settings.db_client_id}