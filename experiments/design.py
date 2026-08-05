from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def stable_seed(identifier: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def balanced_latin_square(items: Sequence[T]) -> list[list[T]]:
    values = list(items)
    count = len(values)
    if count == 0:
        return []

    first_row: list[T] = []
    for index in range(count):
        position = index // 2 if index % 2 == 0 else count - 1 - index // 2
        first_row.append(values[position])

    rows = [
        [values[(values.index(item) + offset) % count] for item in first_row]
        for offset in range(count)
    ]
    if count % 2 == 1:
        rows.extend([list(reversed(row)) for row in rows])
    return rows


def assigned_order(
    items: Sequence[T],
    participant_id: str,
    base_seed: int,
) -> list[T]:
    rows = balanced_latin_square(items)
    if not rows:
        return []
    row_index = stable_seed(participant_id, base_seed) % len(rows)
    return rows[row_index]


def randomized_order(
    items: Sequence[T],
    participant_id: str,
    base_seed: int,
) -> list[T]:
    values = list(items)
    random.Random(stable_seed(participant_id, base_seed)).shuffle(values)
    return values

