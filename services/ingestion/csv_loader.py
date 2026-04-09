"""CSV ingestion service for CloudInsight."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.ingestion.common import (
    IngestionValidationError,
    build_dataset_text_summary,
    build_error_result,
    build_success_result,
    summarize_dataframe,
    validate_file,
)


logger = logging.getLogger(__name__)


class CSVLoader:
    """Load CSV files into a structured payload for analytics and RAG."""

    def __init__(
        self,
        preview_rows: int = 100,
        encoding_candidates: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1"),
    ) -> None:
        self.preview_rows = preview_rows
        self.encoding_candidates = encoding_candidates

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """Load a CSV file and return a normalized tabular result."""
        try:
            path = validate_file(file_path, {".csv"})
        except IngestionValidationError as exc:
            return build_error_result(file_path, "csv", errors=[str(exc)])

        try:
            import pandas as pd
        except ImportError:
            return build_error_result(
                path,
                "csv",
                errors=["Missing dependency 'pandas'. Install it before ingesting CSV files."],
            )

        errors: list[str] = []
        dataframe = None
        selected_encoding = None

        for encoding in self.encoding_candidates:
            try:
                dataframe = pd.read_csv(path, encoding=encoding)
                selected_encoding = encoding
                break
            except UnicodeDecodeError:
                errors.append(f"Unable to decode CSV using '{encoding}'.")
            except pd.errors.EmptyDataError as exc:
                return build_error_result(path, "csv", errors=[f"CSV file is empty: {exc}"])
            except Exception as exc:
                return build_error_result(path, "csv", errors=[f"CSV loading failed: {exc}"])

        if dataframe is None:
            return build_error_result(path, "csv", errors=errors or ["Unable to read CSV file."])

        try:
            table = summarize_dataframe(
                dataframe,
                table_id="csv_data",
                name=path.stem,
                preview_rows=self.preview_rows,
            )
            summary_text = build_dataset_text_summary(path.name, [table])
            sections = [
                {
                    "id": "csv_summary",
                    "type": "dataset_summary",
                    "text": summary_text,
                    "row_count": table["row_count"],
                    "column_count": table["column_count"],
                }
            ]
            warnings = []
            if dataframe.empty:
                warnings.append("CSV file contains headers but no data rows.")

            metadata = {
                "row_count": table["row_count"],
                "column_count": table["column_count"],
                "column_names": table["column_names"],
                "encoding": selected_encoding,
            }

            logger.info("Loaded CSV content from %s", path)
            return build_success_result(
                path,
                "csv",
                metadata=metadata,
                text=summary_text,
                sections=sections,
                tables=[table],
                warnings=warnings,
            )
        except Exception as exc:
            logger.exception("CSV summarization failed for %s", path)
            return build_error_result(path, "csv", errors=[f"CSV summarization failed: {exc}"])

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Alias to align CSV ingestion with extractor-style interfaces."""
        return self.load(file_path)


def load_csv(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for CSV ingestion."""
    return CSVLoader(**kwargs).load(file_path)
