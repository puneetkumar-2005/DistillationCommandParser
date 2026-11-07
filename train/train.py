"""Task 2: train EleutherAI/pythia-70m on teacher replies only.

Sequence-level distillation: each training example is the exact
teacher prompt + command, with the teacher's JSON as the target.
Loss is only on the target tokens (prompt tokens are masked -100).

Usage:
  make train
  make train SUBSET=50
  python train/train.py --config training/config.yaml --subset 50

All hyperparameters come from training/config.yaml; command-line flags
only override them. SUBSET=N trains on the first N records only, and
saves to models/student-v1-subsetN so it never overwrites models/student-v1.
Loss per epoch is written to training/loss.csv (or
training/loss_subsetN.csv for a subset run).

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
import yaml
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
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--subset", type=int, default=0,
                    help="train on the first N records only (0 = all)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = args.data or cfg["data"]
    prompt_path = args.prompt or cfg["prompt"]
    out = args.out or cfg["out"]
    model_name = args.model or cfg["model"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch is not None else cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["learning_rate"]
    max_len = args.max_len if args.max_len is not None else cfg["max_len"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    subset = args.subset
    if subset > 0:
        out = f"{out}-subset{subset}"
        loss_path = f"training/loss_subset{subset}.csv"
    else:
        loss_path = cfg.get("loss_csv", "training/loss.csv")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"torch {torch.__version__}")
    print(f"config: {args.config} subset: {subset} out: {out} loss: {loss_path}")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read().rstrip()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    examples = load_examples(data, prompt_template, tokenizer, max_len)
    if subset > 0:
        examples = examples[:subset]
        print(f"subset: keeping first {len(examples)} examples")
    lengths = [len(e["input_ids"]) for e in examples]
    print(f"tokens: avg {sum(lengths) / len(lengths):.1f} max {max(lengths)}")

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.config.use_cache = False
    model.to(device)
    model.train()

    loader = DataLoader(
        TrainDataset(examples), batch_size=batch, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        generator=torch.Generator().manual_seed(seed),
    )
    total_steps = len(loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=total_steps)
    print(f"steps: {total_steps} ({len(loader)} per epoch, {epochs} epochs)")

    step = 0
    start = time.time()
    epoch_losses = []
    os.makedirs(os.path.dirname(loss_path) or ".", exist_ok=True)
    with open(loss_path, "w", encoding="utf-8") as lf:
        lf.write("epoch,loss\n")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_data in loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                result = model(**batch_data)
                loss = result.loss
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
            avg_loss = total_loss / len(loader)
            epoch_losses.append(avg_loss)
            lf.write(f"{epoch + 1},{avg_loss:.4f}\n")
            lf.flush()
            print(f"epoch {epoch + 1} done, avg loss {avg_loss:.4f}", flush=True)

    print(f"training done in {time.time() - start:.0f}s, final avg loss "
          f"{epoch_losses[-1]:.4f}", flush=True)
    print(f"loss written to {loss_path}", flush=True)

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    config = {
        "base_model": model_name,
        "data": data,
        "n_train": len(examples),
        "epochs": epochs,
        "batch_size": batch,
        "lr": lr,
        "max_len": max_len,
        "seed": seed,
        "subset": subset,
        "torch": torch.__version__,
        "device": device,
    }
    with open(os.path.join(out, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"saved model to {out}")


if __name__ == "__main__":
    main()
