"""Task 1 teacher-labelling worker.

Reads SHARD_INDEX / SHARD_COUNT from the environment:
a command is this worker's if its 0-based line number in
commands_pool.jsonl leaves remainder SHARD_INDEX on division by
SHARD_COUNT. It sends each such command to the teacher (Ollama,
phi3.5, temperature 0, seed 42) with prompts/teacher_prompt.txt, and
writes {"id", "text", "teacher_raw"} -- the teacher's EXACT reply,
unedited -- to datasets/teacher/shard-<SHARD_INDEX>.jsonl in S3.

No record is skipped and no reply is fixed up here: fixing belongs
in clean.py, so the shard objects are an honest audit trail.
"""
import json
import os
import sys
import time

import boto3
import requests
from botocore.config import Config


def env_int(name):
    try:
        return int(os.environ[name])
    except KeyError:
        sys.exit(f"missing required environment variable: {name}")
    except ValueError:
        sys.exit(f"environment variable {name} must be an integer")


def s3_client(endpoint):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def call_teacher(url, model, prompt, timeout=300, retries=3):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 200},
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{url.rstrip('/')}/api/generate",
                              json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as exc:
            last = exc
            print(f"teacher call failed (attempt {attempt}/{retries}): {exc}",
                  flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"teacher call failed after {retries} attempts: {last}")


def main():
    shard_index = env_int("SHARD_INDEX")
    shard_count = env_int("SHARD_COUNT")
    if shard_count <= 0 or not (0 <= shard_index < shard_count):
        sys.exit(f"bad sharding: SHARD_INDEX={shard_index} "
                 f"SHARD_COUNT={shard_count}")

    s3_endpoint = os.environ.get(
        "S3_ENDPOINT", "http://seaweedfs-s3.data.svc.cluster.local:8333")
    bucket = os.environ.get("S3_BUCKET", "datasets")
    input_key = os.environ.get("INPUT_KEY", "commands_pool.jsonl")
    output_key = os.environ.get(
        "OUTPUT_KEY", f"teacher/shard-{shard_index}.jsonl")
    ollama_url = os.environ.get(
        "OLLAMA_URL", "http://ollama.pipeline.svc.cluster.local:11434")
    model = os.environ.get("OLLAMA_MODEL", "phi3.5")
    prompt_path = os.environ.get("PROMPT_PATH", "/prompts/teacher_prompt.txt")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read().rstrip() + "\n"

    s3 = s3_client(s3_endpoint)
    body = s3.get_object(Bucket=bucket, Key=input_key)["Body"].read()
    lines = [ln for ln in body.decode("utf-8").splitlines() if ln.strip()]
    mine = [json.loads(ln) for i, ln in enumerate(lines)
            if i % shard_count == shard_index]
    print(f"shard {shard_index}/{shard_count}: {len(mine)} of "
          f"{len(lines)} commands", flush=True)

    out_path = f"/tmp/shard-{shard_index}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for n, rec in enumerate(mine, 1):
            prompt = (f"{prompt_template}\nCommand: {rec['text']}\nOutput:")
            raw = call_teacher(ollama_url, model, prompt)
            out.write(json.dumps(
                {"id": rec["id"], "text": rec["text"], "teacher_raw": raw},
                ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"shard {shard_index}: {n}/{len(mine)} done", flush=True)

    with open(out_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=output_key, Body=f.read())
    print(f"shard {shard_index}: wrote s3://{bucket}/{output_key} "
          f"({len(mine)} records)", flush=True)


if __name__ == "__main__":
    main()
