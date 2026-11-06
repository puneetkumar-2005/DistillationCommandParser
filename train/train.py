"""Task 2: train EleutherAI/pythia-70m on teacher replies only.

Sequence-level distillation: each training example is the exact
teacher prompt + command, with the teacher's JSON as the target.
Loss is only on the target tokens (prompt tokens are masked -100).

Usage:
  python train/train.py --data data/v1/train1500.jsonl \
      --prompt prompts/teacher_prompt.txt --out models/student-v1

Data: only data/v1/train1500.jsonl (first 1,500 cleaned records).
Never train on student_kit/data/validation.jsonl (evaluation only).
Student: EleutherAI/pythia-70m, fixed. Seed 42.
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_examples(data_path, prompt_template, tokenizer, max_len):
    examples = []
    skipped = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                target = json.loads(rec["teacher_raw"])
            except Exception:
                skipped += 1
                continue
            target_text = json.dumps(target, ensure_ascii=False)
            prompt_text = f"{prompt_template}\nCommand: {rec['text']}\nOutput:"
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + target_ids
            examples.append({"input_ids": input_ids, "labels": labels})
    print(f"loaded {len(examples)} examples, skipped {skipped} (too long or bad JSON)")
    return examples


class TrainDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1/train1500.jsonl")
    ap.add_argument("--prompt", default="prompts/teacher_prompt.txt")
    ap.add_argument("--out", default="models/student-v1")
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"torch {torch.__version__}")

    with open(args.prompt, encoding="utf-8") as f:
        prompt_template = f.read().rstrip()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    examples = load_examples(args.data, prompt_template, tokenizer, args.max_len)
    lengths = [len(e["input_ids"]) for e in examples]
    print(f"tokens: avg {sum(lengths) / len(lengths):.1f} max {max(lengths)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.config.use_cache = False
    model.to(device)
    model.train()

    loader = DataLoader(
        TrainDataset(examples), batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )
    total_steps = len(loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=total_steps)
    print(f"steps: {total_steps} ({len(loader)} per epoch, {args.epochs} epochs)")

    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"step {step}/{total_steps} epoch {epoch + 1} "
                      f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed {time.time() - start:.0f}s", flush=True)
        print(f"epoch {epoch + 1} done, avg loss {total_loss / len(loader):.4f}", flush=True)

    print(f"training done in {time.time() - start:.0f}s, final avg loss "
          f"{total_loss / len(loader):.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    config = {
        "base_model": args.model,
        "data": args.data,
        "n_train": len(examples),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "max_len": args.max_len,
        "seed": args.seed,
        "torch": torch.__version__,
        "device": device,
    }
    with open(os.path.join(args.out, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
