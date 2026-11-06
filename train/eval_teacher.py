"""Task 2: teacher baseline on validation.jsonl only.

Calls the same phi3.5 teacher (Windows Ollama on GPU, temp 0, seed 42)
with the same prompt used for labelling, one command at a time,
then writes predictions in the format score.py expects:
  {"id": "V001", "prediction": "<raw model output text>"}

Usage:
  python train/eval_teacher.py --host http://172.19.208.1:11434 \
      --prompt prompts/teacher_prompt.txt \
      --data student_kit/data/validation.jsonl \
      --out results/task2/teacher_predictions.jsonl

Never train on validation.jsonl. This script only reads it.
Windows Ollama must be running (check with curl first).
"""
import argparse
import json

import requests


def extract_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://172.19.208.1:11434")
    ap.add_argument("--model", default="phi3.5")
    ap.add_argument("--prompt", default="prompts/teacher_prompt.txt")
    ap.add_argument("--data", default="student_kit/data/validation.jsonl")
    ap.add_argument("--out", default="results/task2/teacher_predictions.jsonl")
    args = ap.parse_args()

    with open(args.prompt, encoding="utf-8") as f:
        prompt_template = f.read().rstrip()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"validation examples: {len(rows)}")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            full = f"{prompt_template}\nCommand: {row['text']}\nOutput:"
            r = requests.post(
                f"{args.host}/api/generate",
                json={"model": args.model, "prompt": full, "stream": False,
                      "options": {"temperature": 0, "seed": 42}},
                timeout=120,
            )
            r.raise_for_status()
            raw = r.json()["response"]
            prediction = extract_json(raw)
            out_f.write(json.dumps({"id": row["id"], "prediction": prediction},
                                   ensure_ascii=False) + "\n")
            if i < 5:
                print(f"{row['id']}: {prediction}")
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(rows)} done", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
