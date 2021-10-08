from pathlib import Path
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifier-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifier-metadata.json"


@app.get("/")  # /?q is a url parameter
def read_index(q:Optional[str]=None):
    query = "hello from spam classifier"
    return {'query': query, 'hello': "SPAM ML CLASSIFIER", "BASE_DIR": str(BASE_DIR), "MODEL_DIR": MODEL_DIR.exists(), "MODEL_PATH": MODEL_PATH.exists()}