"""Task 2: evaluate the trained student on validation.jsonl only.

Generates one prediction per validation command with the same
teacher prompt used in training, greedy decoding (temperature 0),
then writes predictions in the format score.py expects:
  {"id": "V001", "prediction": "<raw model output text>"}

Usage:
  python train/eval.py --model models/student-v1 \
      --prompt prompts/teacher_prompt.txt \
      --data student_kit/data/validation.jsonl \
      --out results/task2/predictions.jsonl

Never train on validation.jsonl. This script only reads it.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json(text):
    """Take the first {...} object out of the raw output, if present."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/student-v1")
    ap.add_argument("--prompt", default="prompts/teacher_prompt.txt")
    ap.add_argument("--data", default="student_kit/data/validation.jsonl")
    ap.add_argument("--out", default="results/task2/predictions.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    with open(args.prompt, encoding="utf-8") as f:
        prompt_template = f.read().rstrip()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"validation examples: {len(rows)}")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            prompt_text = f"{prompt_template}\nCommand: {row['text']}\nOutput:"
            inputs = tokenizer(prompt_text, add_special_tokens=False,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
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
