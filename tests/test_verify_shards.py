import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def run_verify(pool, *shards):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_shards.py"),
         str(pool)] + [str(s) for s in shards],
        capture_output=True, text=True)


def test_verify_shards_passes_on_correct_split(tmp_path):
    pool = tmp_path / "pool.jsonl"
    shard0 = tmp_path / "shard-0.jsonl"
    shard1 = tmp_path / "shard-1.jsonl"
    write_jsonl(pool, [{"id": "P1", "text": "a"}, {"id": "P2", "text": "b"}, {"id": "P3", "text": "c"}])
    write_jsonl(shard0, [{"id": "P1", "text": "a"}, {"id": "P3", "text": "c"}])
    write_jsonl(shard1, [{"id": "P2", "text": "b"}])
    result = run_verify(pool, shard0, shard1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing: 0" in result.stdout


def test_verify_shards_fails_when_an_id_is_missing(tmp_path):
    pool = tmp_path / "pool.jsonl"
    shard0 = tmp_path / "shard-0.jsonl"
    shard1 = tmp_path / "shard-1.jsonl"
    write_jsonl(pool, [{"id": "P1", "text": "a"}, {"id": "P2", "text": "b"}, {"id": "P3", "text": "c"}])
    write_jsonl(shard0, [{"id": "P1", "text": "a"}])
    write_jsonl(shard1, [{"id": "P2", "text": "b"}])
    result = run_verify(pool, shard0, shard1)
    assert result.returncode == 1
    assert "missing: 1" in result.stdout
