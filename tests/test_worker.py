import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import worker


def test_env_int_returns_integer(monkeypatch):
    monkeypatch.setenv("SHARD_INDEX", "1")
    assert worker.env_int("SHARD_INDEX") == 1


def test_env_int_missing_variable_exits(monkeypatch):
    monkeypatch.delenv("SHARD_INDEX", raising=False)
    with pytest.raises(SystemExit):
        worker.env_int("SHARD_INDEX")


def test_env_int_non_integer_exits(monkeypatch):
    monkeypatch.setenv("SHARD_INDEX", "two")
    with pytest.raises(SystemExit):
        worker.env_int("SHARD_INDEX")
