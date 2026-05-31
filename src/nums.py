from __future__ import annotations
import json
from pathlib import Path

JSON_FILE_NAME = "pseudoprime_nums.json"
JSON_PATH = Path(__file__).resolve().parents[1] / "data" / JSON_FILE_NAME


def load_pseudoprime_nums() -> dict[str, list[int]]:
    with JSON_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: dict[str, list[int]] = {}
    for name, values in raw.items():
        nums: list[int] = []
        for value in values:
            nums.append(value)
        normalized[name] = nums
    return normalized


PSEUDOPRIME_NUMS = load_pseudoprime_nums()


def reload_pseudoprime_nums() -> dict[str, list[int]]:
    data = load_pseudoprime_nums()
    return data
