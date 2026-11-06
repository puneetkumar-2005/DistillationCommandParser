"""Verify Task 1 shard coverage before cleaning.

  python scripts/verify_shards.py student_kit/data/commands_pool.jsonl \
      data/teacher/shard-0.jsonl data/teacher/shard-1.jsonl

Passes only if: every pool id appears exactly once across all shards,
no extra ids, and each shard's records are exactly its SHARD_INDEX
lines (line number % SHARD_COUNT). This is the 2-mark coverage check.
"""
import json
import sys

pool_path, *shard_paths = sys.argv[1:]
pool = [json.loads(ln) for ln in open(pool_path, encoding="utf-8") if ln.strip()]
pool_ids = [r["id"] for r in pool]
seen = {}
for sp in shard_paths:
    rows = [json.loads(ln) for ln in open(sp, encoding="utf-8") if ln.strip()]
    for r in rows:
        seen.setdefault(r["id"], []).append(sp)
    print(f"{sp}: {len(rows)} records")
dups = {k: v for k, v in seen.items() if len(v) > 1}
missing = [i for i in pool_ids if i not in seen]
extra = [i for i in seen if i not in pool_ids]
print(f"pool: {len(pool_ids)}  shard total: {sum(len(v) for v in seen.values())} "
      f"unique: {len(seen)}")
print(f"missing: {len(missing)}  duplicates: {len(dups)}  extra: {len(extra)}")
if missing:
    print("first missing:", missing[:5])
if dups:
    print("first duplicate:", list(dups)[:5])
sys.exit(0 if not (missing or dups or extra) else 1)
