from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from eegmi.utils import read_json, sha1_hex_from_obj, write_json


@dataclass
class SubjectSplit:
    seed: int
    train_pool_100: list[str]
    train_inner: list[str]
    val_inner: list[str]
    test_9: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "train_pool_100": list(self.train_pool_100),
            "train_inner": list(self.train_inner),
            "val_inner": list(self.val_inner),
            "test_9": list(self.test_9),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubjectSplit":
        return cls(
            seed=int(d["seed"]),
            train_pool_100=list(d["train_pool_100"]),
            train_inner=list(d["train_inner"]),
            val_inner=list(d["val_inner"]),
            test_9=list(d["test_9"]),
        )


def discover_subjects(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    subjects = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("S") and len(p.name) == 4
    )
    return subjects


def _validate_subject_split(split: SubjectSplit, n_train_pool: int, n_test: int, inner_val_count: int) -> None:
    sp = split
    sets = {
        "train_pool_100": set(sp.train_pool_100),
        "train_inner": set(sp.train_inner),
        "val_inner": set(sp.val_inner),
        "test_9": set(sp.test_9),
    }
    assert len(sp.train_pool_100) == n_train_pool, f"Expected {n_train_pool} train_pool subjects"
    assert len(sp.test_9) == n_test, f"Expected {n_test} test subjects"
    assert len(sp.val_inner) == inner_val_count, f"Expected {inner_val_count} val_inner subjects"
    assert sets["train_inner"].isdisjoint(sets["val_inner"]), "train_inner/val_inner overlap"
    assert sets["train_pool_100"] == sets["train_inner"] | sets["val_inner"], "inner split mismatch"
    assert sets["train_pool_100"].isdisjoint(sets["test_9"]), "train_pool/test overlap"


def make_subject_split(
    subjects: list[str],
    seed: int,
    n_train_pool: int = 100,
    n_test: int = 9,
    inner_val_count: int = 10,
) -> SubjectSplit:
    subjects_sorted = sorted(subjects)
    expected_total = n_train_pool + n_test
    if len(subjects_sorted) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} subjects for this policy, found {len(subjects_sorted)}"
        )
    rng = Random(seed)
    shuffled = list(subjects_sorted)
    rng.shuffle(shuffled)
    test_9 = sorted(shuffled[:n_test])
    train_pool_100 = sorted(shuffled[n_test:])

    rng_inner = Random(seed + 1)
    pool_shuffled = list(train_pool_100)
    rng_inner.shuffle(pool_shuffled)
    val_inner = sorted(pool_shuffled[:inner_val_count])
    train_inner = sorted(pool_shuffled[inner_val_count:])

    split = SubjectSplit(
        seed=seed,
        train_pool_100=train_pool_100,
        train_inner=train_inner,
        val_inner=val_inner,
        test_9=test_9,
    )
    _validate_subject_split(split, n_train_pool, n_test, inner_val_count)
    return split


def split_hash(split: SubjectSplit) -> str:
    return sha1_hex_from_obj(split.to_dict())[:16]


def save_subject_split(path: str | Path, split: SubjectSplit) -> None:
    payload = split.to_dict()
    payload["split_hash"] = split_hash(split)
    write_json(path, payload)


def load_subject_split(path: str | Path) -> SubjectSplit:
    payload = read_json(path)
    split = SubjectSplit.from_dict(payload)
    return split
