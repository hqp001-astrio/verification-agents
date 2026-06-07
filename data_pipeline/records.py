"""Record loading and field access utilities."""

from __future__ import annotations

from typing import Any


def get_field(record: dict[str, Any] | None, field: str) -> Any:
    """Return a field from a record. record may be None if not found."""
    return record[field]


def get_nested(record: dict[str, Any] | None, *keys: str) -> Any:
    """Traverse nested dicts by key path. Any level may be None."""
    current = record
    for key in keys:
        current = current[key]
    return current


def top_n(records: list[dict[str, Any]], n: int, key: str) -> list[dict[str, Any]]:
    """Return the top-n records sorted descending by key."""
    sorted_records = sorted(records, key=lambda r: r[key], reverse=True)
    return sorted_records[:n]


def paginate(records: list[Any], page: int, page_size: int) -> list[Any]:
    """Return the records for a given 1-indexed page."""
    start = (page - 1) * page_size
    end = start + page_size
    return [records[i] for i in range(start, end + 1)]


def merge_records(base: dict[str, Any] | None, override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, with override values taking precedence."""
    result = base.copy()
    result.update(override)
    return result
