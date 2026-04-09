"""Shared helpers for CloudInsight ingestion services."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_BLANK_LINE_RE = re.compile(r"\n{3,}")
_INLINE_SPACE_RE = re.compile(r"[^\S\n]+")


class IngestionValidationError(ValueError):
    """Raised when an ingestion request cannot be processed safely."""


def normalize_path(file_path: str | Path) -> Path:
    """Return a resolved path object for downstream processing."""
    return Path(file_path).expanduser().resolve()


def validate_file(file_path: str | Path, expected_suffixes: Iterable[str]) -> Path:
    """Validate that a file exists and matches the expected suffixes."""
    path = normalize_path(file_path)
    suffixes = {suffix.lower() for suffix in expected_suffixes}

    if not path.exists():
        raise IngestionValidationError(f"File not found: {path}")
    if not path.is_file():
        raise IngestionValidationError(f"Path is not a file: {path}")
    if path.suffix.lower() not in suffixes:
        joined = ", ".join(sorted(suffixes))
        raise IngestionValidationError(
            f"Unsupported file extension '{path.suffix or 'unknown'}'. Expected one of: {joined}."
        )

    return path


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines).strip()
    return _BLANK_LINE_RE.sub("\n\n", cleaned)


def join_text_fragments(fragments: Iterable[str]) -> str:
    """Combine cleaned text fragments into a single document body."""
    cleaned_fragments = [clean_text(fragment) for fragment in fragments]
    return "\n\n".join(fragment for fragment in cleaned_fragments if fragment)


def count_words(text: str) -> int:
    """Return a best-effort word count for an extracted text block."""
    return len(text.split()) if text else 0


def base_metadata(path: Path) -> dict[str, Any]:
    """Return file-system level metadata shared by all ingestion results."""
    return {
        "source_extension": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size,
    }


def build_success_result(
    file_path: str | Path,
    file_type: str,
    *,
    metadata: dict[str, Any],
    text: str = "",
    sections: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a normalized success payload for all ingestion services."""
    path = normalize_path(file_path)
    normalized_sections = sections or []
    normalized_tables = tables or []
    combined_metadata = {
        **base_metadata(path),
        **metadata,
        "section_count": len(normalized_sections),
        "table_object_count": len(normalized_tables),
    }

    return {
        "status": "success",
        "file_name": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "detected_type": path.suffix.lower().lstrip("."),
        "metadata": combined_metadata,
        "content": {
            "text": clean_text(text),
            "sections": normalized_sections,
            "tables": normalized_tables,
        },
        "warnings": warnings or [],
        "errors": [],
    }


def build_error_result(
    file_path: str | Path,
    file_type: str,
    *,
    errors: list[str],
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized error payload for all ingestion services."""
    path = normalize_path(file_path)
    resolved_metadata = metadata or {}
    if path.exists():
        resolved_metadata = {**base_metadata(path), **resolved_metadata}

    return {
        "status": "error",
        "file_name": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "detected_type": path.suffix.lower().lstrip("."),
        "metadata": resolved_metadata,
        "content": {"text": "", "sections": [], "tables": []},
        "warnings": warnings or [],
        "errors": errors,
    }


def _to_json_safe_value(value: Any) -> Any:
    """Convert pandas and numpy scalar values into JSON-friendly primitives."""
    if value is None:
        return None

    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    return value


def dataframe_preview_records(dataframe: Any, preview_rows: int) -> list[dict[str, Any]]:
    """Return a JSON-safe row preview for a pandas DataFrame."""
    preview_frame = dataframe.head(preview_rows).copy()
    records: list[dict[str, Any]] = []

    for row in preview_frame.to_dict(orient="records"):
        records.append({str(key): _to_json_safe_value(value) for key, value in row.items()})

    return records


def numeric_summary(dataframe: Any) -> dict[str, dict[str, Any]]:
    """Return descriptive statistics for numeric dataframe columns."""
    numeric_frame = dataframe.select_dtypes(include="number")
    if numeric_frame.empty:
        return {}

    summary_frame = numeric_frame.describe().transpose().round(4)
    return {
        str(column): {str(key): _to_json_safe_value(value) for key, value in stats.items()}
        for column, stats in summary_frame.to_dict(orient="index").items()
    }


def summarize_dataframe(
    dataframe: Any,
    *,
    table_id: str,
    name: str,
    preview_rows: int,
) -> dict[str, Any]:
    """Build structured metadata for a tabular dataset."""
    row_count = int(len(dataframe))
    column_names = [str(column) for column in dataframe.columns]
    preview = dataframe_preview_records(dataframe, preview_rows)

    return {
        "id": table_id,
        "name": name,
        "row_count": row_count,
        "column_count": int(len(column_names)),
        "column_names": column_names,
        "dtypes": {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        "missing_values": {str(column): int(count) for column, count in dataframe.isna().sum().items()},
        "numeric_summary": numeric_summary(dataframe),
        "preview_rows": preview,
        "rows": preview,
    }


def build_dataset_text_summary(file_name: str, tables: list[dict[str, Any]]) -> str:
    """Build a text summary for tabular sources to support downstream retrieval."""
    if not tables:
        return f"{file_name} contains no tabular records."

    lines = [f"{file_name} contains {len(tables)} structured table(s)."]
    for table in tables:
        columns = ", ".join(table.get("column_names", [])[:8])
        lines.append(
            f"{table.get('name', table.get('id', 'table'))}: "
            f"{table.get('row_count', 0)} rows, {table.get('column_count', 0)} columns."
            + (f" Columns include {columns}." if columns else "")
        )
    return " ".join(lines)
