"""Tests for the air-gap guard on corpus construction (PS 2.2.6).

The evaluation workflow must run offline and air-gapped. Corpus construction may fetch
a public dataset once, but the failure this guard prevents is specific and nasty: on a
judge box with no route to cs231n.stanford.edu, `urlretrieve` does not fail fast -- it
stalls on connect retries. A 20k-model run that stalls on the first corpus call is
worse than one that refuses, so `CVIAF_OFFLINE=1` must turn an unbounded stall into a
message that names the cache that is missing.

Each test below is a failure mode, not a code path:

  * offline + empty cache must raise, and must not litter the cache directory first;
  * offline + populated cache must proceed (offline is a policy, not a prohibition);
  * unset means "allowed", so existing pipelines cannot break on a typo;
  * the message must name the cache directory and what it looked for, because a
    refusal that does not say what to populate is just a stall with better manners.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.cifar import (  # noqa: E402
    CIFAR_DIRNAME,
    OfflineError,
    ensure_cifar10,
    offline_requested,
)


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    (" 1 ", True),
    ("", False), ("0", False), ("false", False), ("no", False), ("off", False),
    ("maybe", False),
])
def test_offline_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("CVIAF_OFFLINE", value)
    assert offline_requested() is expected


def test_offline_is_off_when_unset(monkeypatch):
    monkeypatch.delenv("CVIAF_OFFLINE", raising=False)
    assert offline_requested() is False


def test_offline_with_empty_cache_raises_before_touching_the_filesystem(monkeypatch,
                                                                       tmp_path):
    monkeypatch.setenv("CVIAF_OFFLINE", "1")
    cache = str(tmp_path / "cache")
    assert not os.path.exists(cache)
    with pytest.raises(OfflineError) as excinfo:
        ensure_cifar10(cache_dir=cache, verbose=False)
    message = str(excinfo.value)
    # The refusal has to be actionable: which cache, and what was looked for.
    assert "cache" in message
    assert "cifar-10" in message.lower()
    assert "CVIAF_OFFLINE" in message
    # No half-built cache directory: a stall replaced by a refusal must not also
    # leave the next run believing the dataset is being prepared.
    assert not os.path.exists(cache)


def test_offline_with_populated_cache_proceeds(monkeypatch, tmp_path):
    monkeypatch.setenv("CVIAF_OFFLINE", "1")
    root = tmp_path / "cache" / CIFAR_DIRNAME
    root.mkdir(parents=True)
    (root / "data_batch_1").write_bytes(b"fixture")
    got = ensure_cifar10(cache_dir=str(tmp_path / "cache"), verbose=False)
    assert got == str(root)


def test_unset_flag_with_populated_cache_does_not_fetch(monkeypatch, tmp_path):
    """Offline is a policy; an already-cached dataset must not be re-fetched either way."""
    monkeypatch.delenv("CVIAF_OFFLINE", raising=False)
    root = tmp_path / "cache" / CIFAR_DIRNAME
    root.mkdir(parents=True)
    (root / "data_batch_1").write_bytes(b"fixture")
    assert ensure_cifar10(cache_dir=str(tmp_path / "cache"), verbose=False) == str(root)
