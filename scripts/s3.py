"""Tiny S3 helper for the local SeaweedFS endpoint (no AWS account).

Env: S3_ENDPOINT (default http://localhost:8333 via port-forward),
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (dummy "local" values are
fine: local SeaweedFS auth is off).
  python scripts/s3.py mb datasets models
  python scripts/s3.py ls datasets
  python scripts/s3.py upload datasets student_kit/data/commands_pool.jsonl commands_pool.jsonl
  python scripts/s3.py download datasets teacher/shard-0.jsonl data/teacher/shard-0.jsonl
"""
import os
import sys

import boto3
from botocore.config import Config


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:8333"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def main():
    c, cmd, a = client(), sys.argv[1], sys.argv[2:]
    if cmd == "mb":
        for b in a:
            c.create_bucket(Bucket=b)
            print(f"created bucket: {b}")
    elif cmd == "ls":
        r = c.list_objects_v2(Bucket=a[0], Prefix=a[1] if len(a) > 1 else "")
        for o in r.get("Contents", []):
            print(f"{o['Key']}  {o['Size']} bytes")
        if not r.get("Contents"):
            print("(empty)")
    elif cmd == "upload":
        c.upload_file(a[1], a[0], a[2])
        print(f"uploaded {a[1]} -> s3://{a[0]}/{a[2]}")
    elif cmd == "download":
        os.makedirs(os.path.dirname(a[2]) or ".", exist_ok=True)
        c.download_file(a[0], a[1], a[2])
        print(f"downloaded s3://{a[0]}/{a[1]} -> {a[2]}")
    else:
        sys.exit("usage: mb|ls|upload|download ...")


if __name__ == "__main__":
    main()
