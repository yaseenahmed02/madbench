"""Load Track B workload YAML files into a flat, ordered question list.

A workload file maps SQLyzr category keys (``c1`` .. ``c6``) to lists of
questions. Callers may pass several files and get one combined list, which
lets users add their own question files next to ``questions.yaml``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_CATEGORY_KEY = re.compile(r"c[1-6]")


class WorkloadError(ValueError):
    """Raised when a workload file is empty, malformed, or ids collide."""


def load_workload(paths: list[Path]) -> list[dict[str, Any]]:
    """Return every question from ``paths``, in file order then YAML order.

    Args:
        paths: One or more workload YAML files.

    Returns:
        Dicts with ``id``, ``category``, ``target_subcat``, ``question`` and
        ``gold_sql``.

    Raises:
        WorkloadError: A file holds no questions, or an id appears twice.
    """
    questions: list[dict[str, Any]] = []
    for path in paths:
        loaded = _load_one(path)
        if not loaded:
            raise WorkloadError(f"no c1-c6 questions found in {path}")
        questions.extend(loaded)
    _reject_duplicate_ids(questions)
    return questions


def _load_one(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text()) or {}
    questions = []
    for category, entries in payload.items():
        if not _CATEGORY_KEY.fullmatch(str(category)):
            continue
        for entry in entries or []:
            questions.append({
                "id": entry["id"],
                "category": category,
                "target_subcat": entry.get("target_subcat", ""),
                "question": entry["question"],
                "gold_sql": entry["gold_sql"].strip(),
            })
    return questions


def _reject_duplicate_ids(questions: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates = sorted({q["id"] for q in questions
                         if q["id"] in seen or seen.add(q["id"])})
    if duplicates:
        raise WorkloadError(f"duplicate question ids: {duplicates}")
