"""Tests for the provenance ledger.

The point of the ledger is that a number can be re-derived from it, so these tests
are written as the attacks on that claim: edit a record, delete one, reorder two,
recompute a hash to cover the edit, or change an artefact after recording it. Each
must be caught, and caught at a named index.
"""
from __future__ import annotations

import json
import os

import pytest

from cviaf.lab.provenance_ledger import (GENESIS, LEDGER_SCHEMA, append, canonical,
                                        describe_artifact, digest_of_ledger,
                                        entry_digest, file_digest, init_ledger, main,
                                        make_record, read_ledger, tree_digest, verify)


def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def make_ledger(tmp_path, n=3, artefacts=None):
    path = os.path.join(str(tmp_path), "ledger.jsonl")
    init_ledger(path, note="test")
    for i in range(n):
        append(path, f"event{i}", command=f"cmd {i}",
               payload={"i": i, "ratio": i / 4}, artefacts=artefacts)
    return path


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_init_then_append_produces_a_valid_chain(tmp_path):
    path = make_ledger(tmp_path)
    report = verify(path)
    assert report["valid"], report["problems"]
    assert report["n_entries"] == 3
    records = read_ledger(path)
    assert records[0]["schema"] == LEDGER_SCHEMA
    assert records[1]["prev_hash"] == GENESIS
    assert records[2]["prev_hash"] == records[1]["entry_hash"]
    assert [r["seq"] for r in records[1:]] == [0, 1, 2]


def test_records_the_commit_they_were_written_at(tmp_path):
    path = make_ledger(tmp_path, 1)
    commit = read_ledger(path)[1]["git_commit"]
    assert commit is None or (len(commit) == 40 and commit.isalnum()), commit


def test_the_line_on_disk_is_the_canonical_record(tmp_path):
    """Sorting and separators are fixed on disk, so two writers cannot disagree."""
    path = make_ledger(tmp_path, 2)
    line = read_lines(path)[1]
    record = json.loads(line)
    assert line == canonical(record).decode()          # byte-identical re-encoding
    assert record["entry_hash"] == entry_digest(record)
    assert line == canonical(json.loads(line)).decode()


def test_ledger_digest_changes_only_when_the_ledger_does(tmp_path):
    path = make_ledger(tmp_path, 2)
    before = digest_of_ledger(path)
    assert digest_of_ledger(path) == before
    append(path, "later")
    assert digest_of_ledger(path) != before


def test_init_refuses_to_clobber_an_existing_ledger(tmp_path):
    path = make_ledger(tmp_path, 1)
    with pytest.raises(FileExistsError):
        init_ledger(path)


def test_append_creates_the_ledger_if_absent(tmp_path):
    path = os.path.join(str(tmp_path), "new", "ledger.jsonl")
    append(path, "first")
    assert verify(path)["valid"]


def test_payload_values_round_trip(tmp_path):
    path = os.path.join(str(tmp_path), "l.jsonl")
    append(path, "battery", payload={"asset_tpr": 0.1458, "rejected": 7,
                                     "kinds": ["weight_tamper"]})
    payload = read_ledger(path)[1]["payload"]
    assert payload["asset_tpr"] == 0.1458 and payload["rejected"] == 7
    assert payload["kinds"] == ["weight_tamper"]


# --------------------------------------------------------------------------- #
# tampering
# --------------------------------------------------------------------------- #

def test_edited_record_is_caught_at_its_index(tmp_path):
    """The headline claim: a payload cannot be edited after the fact."""
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    record = json.loads(lines[2])
    record["payload"]["asset_tpr"] = 0.99
    lines[2] = json.dumps(record, sort_keys=True)
    write_lines(path, lines)

    report = verify(path)
    assert not report["valid"] and report["broken_at"] == 1
    assert "does not hash to its recorded entry_hash" in report["problems"][0]


def test_edit_plus_recomputed_hash_is_caught_by_the_next_link(tmp_path):
    """An attacker who re-hashes the edited record still breaks the chain."""
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    record = json.loads(lines[1])
    record["payload"]["i"] = 999
    record["entry_hash"] = entry_digest(record)
    lines[1] = json.dumps(record, sort_keys=True)
    write_lines(path, lines)

    report = verify(path)
    assert not report["valid"] and report["broken_at"] == 1
    assert "does not link" in report["problems"][0]


def test_deleted_record_is_caught_as_a_sequence_gap(tmp_path):
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    del lines[2]
    write_lines(path, lines)
    report = verify(path)
    assert not report["valid"] and report["broken_at"] == 1
    assert "seq 2" in report["problems"][0]


def test_reordered_records_are_caught(tmp_path):
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    lines[1], lines[2] = lines[2], lines[1]
    write_lines(path, lines)
    assert not verify(path)["valid"]


def test_duplicated_record_is_caught(tmp_path):
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    lines.append(lines[-1])
    write_lines(path, lines)
    report = verify(path)
    assert not report["valid"] and report["broken_at"] == 3


def test_unknown_schema_is_rejected(tmp_path):
    path = make_ledger(tmp_path, 1)
    lines = read_lines(path)
    header = json.loads(lines[0])
    header["schema"] = "cviaf.something-else.v1"
    lines[0] = json.dumps(header, sort_keys=True)
    write_lines(path, lines)
    assert not verify(path)["valid"]


def test_appending_to_a_broken_ledger_is_refused(tmp_path):
    path = make_ledger(tmp_path)
    lines = read_lines(path)
    lines[1] = lines[1].replace('"i":0', '"i":42')
    assert json.loads(lines[1])["payload"]["i"] == 42
    write_lines(path, lines)
    with pytest.raises(ValueError, match="refusing to append"):
        append(path, "after-tamper")


def test_missing_ledger_reports_rather_than_raises(tmp_path):
    report = verify(os.path.join(str(tmp_path), "nope.jsonl"))
    assert not report["valid"] and report["n_entries"] == 0
    assert "no such ledger" in report["problems"][0]


def test_nan_payload_is_refused_at_write_time(tmp_path):
    """NaN cannot round-trip through JSON; it must never enter the ledger."""
    with pytest.raises(ValueError):
        make_record(0, GENESIS, "bad", payload={"score": float("nan")})
    with pytest.raises(ValueError):
        make_record(0, GENESIS, "bad", payload={"score": float("inf")})


def test_missing_artefact_is_refused_at_write_time(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_record(0, GENESIS, "bad", artefacts={"out": "/nonexistent/file.json"})


# --------------------------------------------------------------------------- #
# artefacts: re-deriving the claim
# --------------------------------------------------------------------------- #

def test_unchanged_artefacts_verify(tmp_path):
    target = os.path.join(str(tmp_path), "result.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"asset_tpr": 0.1458}, fh)
    path = make_ledger(tmp_path, 1, artefacts={"out:result": target})
    report = verify(path, check_artefacts=True)
    assert report["valid"] and report["artefacts_checked"] == 1
    assert report["artefacts_drifted"] == []


def test_changed_artefact_is_reported_as_drift(tmp_path):
    """The stale-headline bug: a number that no longer describes the file."""
    target = os.path.join(str(tmp_path), "result.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"asset_tpr": 0.1875}, fh)
    path = make_ledger(tmp_path, 1, artefacts={"out:result": target})
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"asset_tpr": 0.1458}, fh)

    report = verify(path, check_artefacts=True)
    assert not report["valid"]
    assert report["artefacts_drifted"][0]["role"] == "out:result"
    assert "changed since it was recorded" in report["problems"][0]


def test_deleted_artefact_is_reported(tmp_path):
    target = os.path.join(str(tmp_path), "result.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"x": 1}, fh)
    path = make_ledger(tmp_path, 1, artefacts={"out:result": target})
    os.remove(target)
    report = verify(path, check_artefacts=True)
    assert not report["valid"] and "is gone" in report["problems"][0]


def test_artefact_paths_can_be_resolved_relative_to_a_base_dir(tmp_path):
    corpus = os.path.join(str(tmp_path), "corpus")
    os.makedirs(corpus)
    target = os.path.join(corpus, "manifest.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"model_id": "m0"}, fh)
    path = os.path.join(str(tmp_path), "l.jsonl")
    init_ledger(path)
    append(path, "corpus", artefacts={"in:corpus": "corpus/manifest.json"},
           base_dir=str(tmp_path))
    assert verify(path, check_artefacts=True, base_dir=str(tmp_path))["valid"]
    meta = read_ledger(path)[1]["artefacts"]["in:corpus"]
    assert meta["path"] == "corpus/manifest.json"      # readable as written
    assert os.path.isabs(meta["resolved"])            # but resolvable anywhere


def make_tree(root):
    os.makedirs(os.path.join(root, "m0"), exist_ok=True)
    os.makedirs(os.path.join(root, "m1"), exist_ok=True)
    for name, body in (("m0/manifest.json", '{"model_id":"m0"}'),
                       ("m1/manifest.json", '{"model_id":"m1"}'),
                       ("registry.jsonl", '{"model_id":"m0"}\n')):
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return root


def test_tree_digest_binds_a_corpus(tmp_path):
    """A corpus is the input that silently changed; the ledger must name it."""
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    first = tree_digest(root)
    assert first["kind"] == "tree" and first["files"] == 3
    assert tree_digest(root)["sha256"] == first["sha256"]      # stable

    with open(os.path.join(root, "m1", "manifest.json"), "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert tree_digest(root)["sha256"] != first["sha256"]      # edited


def test_tree_digest_notices_added_removed_and_renamed_files(tmp_path):
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    base = tree_digest(root)["sha256"]
    with open(os.path.join(root, "m1", "weights.npz"), "wb") as fh:
        fh.write(b"w")
    added = tree_digest(root)["sha256"]
    assert added != base
    os.rename(os.path.join(root, "m1", "weights.npz"),
              os.path.join(root, "m1", "weights2.npz"))
    assert tree_digest(root)["sha256"] != added
    os.remove(os.path.join(root, "m1", "weights2.npz"))
    assert tree_digest(root)["sha256"] == base


def test_describe_artifact_dispatches_on_file_or_directory(tmp_path):
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    assert describe_artifact(root)["kind"] == "tree"
    assert describe_artifact(os.path.join(root, "registry.jsonl"))["kind"] == "file"


def test_excluded_outputs_do_not_make_a_corpus_go_stale(tmp_path):
    """A battery writing its result next to the models must not look like corpus drift."""
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    path = os.path.join(str(tmp_path), "l.jsonl")
    init_ledger(path)
    append(path, "battery", artefacts={"in:corpus": root},
           excludes={"in:corpus": ("*.log", "compare_*.json")})
    baseline = tree_digest(root)["sha256"]
    with open(os.path.join(root, "compare_73.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    with open(os.path.join(root, "compare_73.log"), "w", encoding="utf-8") as fh:
        fh.write("log\n")
    assert verify(path, check_artefacts=True)["valid"]
    # but a model file appearing in the corpus still changes it
    with open(os.path.join(root, "m1", "manifest.json"), "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert tree_digest(root, exclude=("*.log", "compare_*.json"))["sha256"] != baseline
    assert not verify(path, check_artefacts=True)["valid"]


def test_the_exclude_definition_travels_with_the_record(tmp_path):
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    path = os.path.join(str(tmp_path), "l.jsonl")
    init_ledger(path)
    append(path, "battery", artefacts={"in:corpus": root},
           excludes={"in:corpus": ("*.log",)})
    meta = read_ledger(path)[1]["artefacts"]["in:corpus"]
    assert meta["exclude"] == ["*.log"] and meta["kind"] == "tree"
    assert verify(path, check_artefacts=True)["valid"]


def test_a_changed_corpus_is_reported_as_drift(tmp_path):
    """The stale-headline failure, exactly: a number bound to a corpus that moved."""
    root = make_tree(os.path.join(str(tmp_path), "corpus"))
    path = os.path.join(str(tmp_path), "l.jsonl")
    init_ledger(path)
    append(path, "battery", artefacts={"in:corpus": root})
    assert verify(path, check_artefacts=True)["valid"]
    with open(os.path.join(root, "m1", "manifest.json"), "a", encoding="utf-8") as fh:
        fh.write("\n")
    report = verify(path, check_artefacts=True)
    assert not report["valid"] and report["artefacts_drifted"][0]["role"] == "in:corpus"


def test_file_digest_records_size_so_a_truncation_is_visible(tmp_path):
    target = os.path.join(str(tmp_path), "f.bin")
    with open(target, "wb") as fh:
        fh.write(b"a" * 10)
    first = file_digest(target)
    with open(target, "wb") as fh:
        fh.write(b"a" * 9)
    second = file_digest(target)
    assert first["sha256"] != second["sha256"] and second["bytes"] == 9


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_init_append_verify_round_trip(tmp_path, capsys):
    path = os.path.join(str(tmp_path), "l.jsonl")
    target = os.path.join(str(tmp_path), "out.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"asset_tpr": 0.1458}, fh)

    assert main(["init", "--path", path, "--note", "lane"]) == 0
    assert main(["append", "--path", path, "--event", "battery",
                 "--command", "python scripts/battery_model_attacks.py",
                 "--output", f"result={target}",
                 "--payload", "asset_tpr=0.1458"]) == 0
    assert main(["verify", "--path", path, "--check-artefacts"]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out and "1 artefact(s) re-hashed" in out
    assert read_ledger(path)[1]["payload"]["asset_tpr"] == 0.1458


def test_cli_verify_fails_on_drift(tmp_path, capsys):
    path = os.path.join(str(tmp_path), "l.jsonl")
    target = os.path.join(str(tmp_path), "out.json")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("{}")
    main(["init", "--path", path])
    main(["append", "--path", path, "--event", "e", "--output", f"r={target}"])
    with open(target, "w", encoding="utf-8") as fh:
        fh.write('{"changed": true}')
    capsys.readouterr()
    assert main(["verify", "--path", path, "--check-artefacts"]) == 1
    assert "INVALID" in capsys.readouterr().out


def test_cli_verify_reports_a_missing_ledger(tmp_path, capsys):
    assert main(["verify", "--path", os.path.join(str(tmp_path), "nope")]) == 1
    capsys.readouterr()
