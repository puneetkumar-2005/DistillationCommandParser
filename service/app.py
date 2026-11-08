"""Task 3: command-parser inference service.

Endpoints (port 8000):
  POST /parse   {"text": "..."} -> 200 + the four-key JSON, nothing else.
                  Headers: X-Model-Version, X-Served-By (Pod name).
                  Invalid model output -> 500 {"error": "invalid_model_output"}.
  GET  /healthz -> 200 while the process is alive.
  GET  /readyz  -> 503 until the model is loaded, 200 after that.

Config, all from environment variables:
  MODEL_URI       e.g. s3://models/v1  (bucket models, prefix v1)
  MODEL_VERSION   e.g. v1
  S3_ENDPOINT     e.g. http://seaweedfs-s3.data.svc.cluster.local:8333
  MAX_NEW_TOKENS  e.g. 64
  FAULT_INJECT    1 replaces the model output with "not json" before checking.
Credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) come from the
environment only -- in Kubernetes, from a Secret, never from the code.

At start-up the service downloads the model and tokenizer from object
storage into /tmp/model, then loads them on CPU.
"""
import json
import os
import socket
import threading

import boto3
import torch
from botocore.config import Config
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)

MODEL_URI = os.environ.get("MODEL_URI", "s3://models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0") == "1"
MODEL_DIR = "/tmp/model"

app = FastAPI()
model = None
tokenizer = None
model_loaded = False
prompt_template = ""
schema = {"keys": ["intent", "date", "time", "place"],
          "intents": ["alarm_set", "alarm_query", "calendar_set",
                      "calendar_query", "calendar_remove", "weather_query",
                      "datetime_query", "recommendation_events"]}


def load_text_file(*paths):
    for p in paths:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    return None


def download_model():
    """Download every object under MODEL_URI into MODEL_DIR."""
    if not MODEL_URI.startswith("s3://"):
        raise ValueError(f"MODEL_URI must start with s3://, got {MODEL_URI}")
    rest = MODEL_URI[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    client = boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    if not keys:
        raise RuntimeError(f"no objects found at {MODEL_URI}")
    for key in keys:
        name = key[len(prefix):] if key.startswith(prefix) else os.path.basename(key)
        if not name or name.endswith("/"):
            continue
        dest = os.path.join(MODEL_DIR, name)
        os.makedirs(os.path.dirname(dest) or MODEL_DIR, exist_ok=True)
        client.download_file(bucket, key, dest)
        print(f"downloaded s3://{bucket}/{key} -> {dest}", flush=True)
    return MODEL_DIR


def load_model():
    global model, tokenizer, model_loaded, prompt_template, schema
    prompt = load_text_file("service/prompt.txt", "prompts/teacher_prompt.txt",
                            "/app/prompt.txt")
    if prompt is None:
        raise RuntimeError("teacher prompt file not found")
    prompt_template = prompt.rstrip()
    schema_text = load_text_file("service/schema.json",
                                 "student_kit/data/schema.json",
                                 "/app/schema.json")
    if schema_text:
        schema = json.loads(schema_text)
    model_dir = download_model()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    model_loaded = True
    print("model loaded", flush=True)


@app.on_event("startup")
def startup():
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()


class ParseRequest(BaseModel):
    text: str


def extract_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def valid_output(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != set(schema["keys"]):
        return False
    if obj["intent"] not in schema["intents"]:
        return False
    for slot in ("date", "time", "place"):
        if obj[slot] is not None and not isinstance(obj[slot], str):
            return False
    return True


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if model_loaded:
        return {"status": "ready"}
    return JSONResponse({"status": "loading"}, status_code=503)


@app.post("/parse")
def parse(req: ParseRequest, response: Response):
    if not model_loaded:
        return JSONResponse({"error": "model_not_ready"}, status_code=503)
    full = f"{prompt_template}\nCommand: {req.text}\nOutput:"
    inputs = tokenizer(full, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if FAULT_INJECT:
        raw = "not json"
    text = extract_json(raw)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return JSONResponse({"error": "invalid_model_output"}, status_code=500)
    if not valid_output(obj):
        return JSONResponse({"error": "invalid_model_output"}, status_code=500)
    response.headers["X-Model-Version"] = MODEL_VERSION
    response.headers["X-Served-By"] = os.environ.get("HOSTNAME", socket.gethostname())
    return obj


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
