"""Task 1 cleaning: shard outputs -> datasets/v1/train.jsonl.

Usage:
  python worker/clean.py --shards data/teacher/shard-0.jsonl data/teacher/shard-1.jsonl \
      --out data/v1/train.jsonl
  # then: sha256sum data/v1/train.jsonl > data/v1/train.sha256
  # and upload both to s3://datasets/v1/ (see README_TASK1.md, Step 9).

Rules, in order, one count per record (so the counts balance):
  (a) teacher_raw is not valid JSON
  (b) parsed JSON is not a dict with exactly intent, date, time, place
  (c) intent is not one of the eight
  (d) a slot is neither null nor a string that appears in the command
      (case-insensitive substring check)
  duplicates: after (a)-(d) pass, a command already seen (normalised
      lower-case, collapsed whitespace) -- keep the smallest id.
Records are processed in id order, so "smallest id" is automatic.
Output keeps the shard record shape {id, text, teacher_raw} in id order.
"""
import argparse
import hashlib
import json
import sys

KEYS = {"intent", "date", "time", "place"}
INTENTS = {"alarm_set", "alarm_query", "calendar_set", "calendar_query",
           "calendar_remove", "weather_query", "datetime_query",
           "recommendation_events"}


def norm_text(s):
    return " ".join(str(s).strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", default=None,
                    help="write counts JSON here (default: <out>.counts.json)")
    args = ap.parse_args()

    records = []
    for path in args.shards:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda r: r.get("id", ""))

    counts = {"input": len(records), "a_invalid_json": 0,
              "b_wrong_keys": 0, "c_bad_intent": 0, "d_bad_slot": 0,
              "duplicates": 0, "kept": 0}
    seen = set()
    kept = []
    for rec in records:
        raw = rec.get("teacher_raw")
        try:
            obj = json.loads(raw.strip()) if isinstance(raw, str) else None
            if obj is None:
                raise ValueError("teacher_raw is not a JSON string")
        except Exception:
            counts["a_invalid_json"] += 1
            continue
        if not isinstance(obj, dict) or set(obj.keys()) != KEYS:
            counts["b_wrong_keys"] += 1
            continue
        if obj["intent"] not in INTENTS:
            counts["c_bad_intent"] += 1
            continue
        cmd = str(rec.get("text", "")).lower()
        bad_slot = False
        for slot in ("date", "time", "place"):
            v = obj[slot]
            if v is None:
                continue
            if not isinstance(v, str) or v.strip().lower() not in cmd:
                bad_slot = True
                break
        if bad_slot:
            counts["d_bad_slot"] += 1
            continue
        key = norm_text(rec.get("text", ""))
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        kept.append(rec)
        counts["kept"] += 1

    total = (counts["a_invalid_json"] + counts["b_wrong_keys"]
             + counts["c_bad_intent"] + counts["d_bad_slot"]
             + counts["duplicates"] + counts["kept"])
    assert total == counts["input"], f"counts do not balance: {total} != {counts['input']}"

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    counts_path = args.counts or (args.out + ".counts.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts, indent=2))
    print(f"sha256: {sha}  (write this to train.sha256 as "
          f"'{sha}  train.jsonl')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
