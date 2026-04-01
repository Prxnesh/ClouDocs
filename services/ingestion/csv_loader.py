"""CSV ingestion utilities for CloudInsight v2."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


class CSVLoader:
    """Load CSV files into a standardized ingestion payload."""

    SUPPORTED_SUFFIXES = {".csv"}

    def __init__(self, *, chunksize: int = 10000, preview_rows: int = 50) -> None:
        self.chunksize = chunksize
        self.preview_rows = preview_rows

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """Load CSV content and return a standardized payload."""
        path = Path(file_path).expanduser()
        detected_type = self._detect_file_type(path)

        try:
            self._validate(path, detected_type)
            delimiter = self._detect_delimiter(path)

            row_count = 0
            chunks_processed = 0
            preview_frames: list[pd.DataFrame] = []
            header_frame = pd.read_csv(path, nrows=0, sep=delimiter)
            columns = header_frame.columns.tolist()
            dtypes: dict[str, str] = {}
            warnings: list[str] = []

            for chunk in pd.read_csv(path, chunksize=self.chunksize, sep=delimiter):
                chunks_processed += 1
                row_count += len(chunk)

                if not dtypes:
                    dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

                preview_collected = sum(len(frame) for frame in preview_frames)
                if preview_collected < self.preview_rows:
                    remaining = self.preview_rows - preview_collected
                    preview_frames.append(chunk.head(remaining).copy())

            if row_count == 0:
                warnings.append("CSV file was parsed successfully but contains no data rows.")

            preview_df = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame(columns=columns)
            dataset_summary = self._build_dataset_summary(path, row_count, columns, dtypes)
            metadata = {
                "file_size_bytes": path.stat().st_size,
                "row_count": row_count,
                "column_count": len(columns),
                "delimiter": delimiter,
                "chunks_processed": chunks_processed,
                "preview_row_count": len(preview_df),
                "truncated": row_count > len(preview_df),
            }
            content = {
                "text": dataset_summary,
                "sections": [
                    {
                        "id": "dataset_1",
                        "type": "dataset",
                        "name": path.stem,
                        "text": dataset_summary,
                        "columns": columns,
                    }
                ],
                "tables": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "name": path.stem,
                        "columns": columns,
                        "dtypes": dtypes,
                        "preview_rows": preview_df.to_dict(orient="records"),
                    }
                ],
            }
            return self._success_response(path, detected_type, metadata, content, warnings)
        except Exception as exc:  # pragma: no cover - exercised through integration
            return self._error_response(path, detected_type, exc)

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Alias to align CSV ingestion with extractor-style interfaces."""
        return self.load(file_path)

    def _build_dataset_summary(
        self,
        path: Path,
        row_count: int,
        columns: list[str],
        dtypes: dict[str, str],
    ) -> str:
        column_summary = ", ".join(f"{name} ({dtype})" for name, dtype in dtypes.items()) or "No columns detected"
        return f"Dataset '{path.stem}' contains {row_count} rows and {len(columns)} columns: {column_summary}."

    def _detect_delimiter(self, path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            sample = handle.read(4096)

        if not sample.strip():
            return ","

        try:
            return csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            return ","

    def _detect_file_type(self, path: Path) -> str:
        return path.suffix.lower().lstrip(".") or "unknown"

    def _validate(self, path: Path, detected_type: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file path, received: {path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type for CSV loader: {detected_type}")

    def _success_response(
        self,
        path: Path,
        detected_type: str,
        metadata: dict[str, Any],
        content: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_type": "csv",
            "detected_type": detected_type,
            "metadata": metadata,
            "content": content,
            "warnings": warnings,
            "errors": [],
        }

    def _error_response(self, path: Path, detected_type: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": "error",
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_type": "csv",
            "detected_type": detected_type,
            "metadata": {},
            "content": {"text": "", "sections": [], "tables": []},
            "warnings": [],
            "errors": [str(exc)],
        }


def load_csv(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for CSV ingestion."""
    return CSVLoader(**kwargs).load(file_path)
