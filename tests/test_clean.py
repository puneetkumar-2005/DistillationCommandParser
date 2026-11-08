import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import clean


def test_norm_text_collapses_whitespace_and_lowercases():
    assert clean.norm_text("  Set   An ALARM   for  Tomorrow ") == "set an alarm for tomorrow"


def test_keys_and_intents_are_the_assignment_set():
    assert clean.KEYS == {"intent", "date", "time", "place"}
    assert len(clean.INTENTS) == 8
    assert "alarm_set" in clean.INTENTS
    assert "recommendation_events" in clean.INTENTS


def test_clean_counts_balance_on_one_record_per_class(tmp_path):
    shard = tmp_path / "shard.jsonl"
    out = tmp_path / "train.jsonl"
    counts_path = tmp_path / "counts.json"
    records = [
        {"id": "C001", "text": "set an alarm for tomorrow at seven thirty am",
         "teacher_raw": '{"intent": "alarm_set", "date": "tomorrow", "time": "seven thirty am", "place": null}'},
        {"id": "C002", "text": "what time is it",
         "teacher_raw": "not json at all"},
        {"id": "C003", "text": "set an alarm",
         "teacher_raw": '{"intent": "alarm_set", "date": null, "time": null}'},
        {"id": "C004", "text": "book a flight to the moon",
         "teacher_raw": '{"intent": "fly_to_moon", "date": null, "time": null, "place": null}'},
        {"id": "C005", "text": "set an alarm for tomorrow",
         "teacher_raw": '{"intent": "alarm_set", "date": "next year", "time": null, "place": null}'},
        {"id": "C006", "text": "set an alarm for tomorrow at seven thirty am",
         "teacher_raw": '{"intent": "alarm_set", "date": "tomorrow", "time": "seven thirty am", "place": null}'},
    ]
    shard.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "worker" / "clean.py"),
         "--shards", str(shard), "--out", str(out), "--counts", str(counts_path)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    assert counts == {"input": 6, "a_invalid_json": 1, "b_wrong_keys": 1,
                      "c_bad_intent": 1, "d_bad_slot": 1, "duplicates": 1, "kept": 1}
    kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["id"] for r in kept] == ["C001"]
