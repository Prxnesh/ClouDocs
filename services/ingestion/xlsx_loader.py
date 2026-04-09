"""XLSX ingestion service for CloudInsight."""

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


class XLSXLoader:
    """Load workbook sheets into a normalized multi-table payload."""

    def __init__(self, preview_rows: int = 100) -> None:
        self.preview_rows = preview_rows

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """Load an XLSX workbook and surface sheet-level table metadata."""
        try:
            path = validate_file(file_path, {".xlsx"})
        except IngestionValidationError as exc:
            return build_error_result(file_path, "xlsx", errors=[str(exc)])

        try:
            import pandas as pd
        except ImportError:
            return build_error_result(
                path,
                "xlsx",
                errors=["Missing dependency 'pandas'. Install it before ingesting XLSX files."],
            )

        try:
            workbook = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            return build_error_result(path, "xlsx", errors=[f"XLSX loading failed: {exc}"])

        try:
            tables: list[dict[str, Any]] = []
            sections: list[dict[str, Any]] = []
            total_rows = 0
            max_columns = 0

            for sheet_name, dataframe in workbook.items():
                table = summarize_dataframe(
                    dataframe,
                    table_id=sheet_name,
                    name=sheet_name,
                    preview_rows=self.preview_rows,
                )
                tables.append(table)
                total_rows += table["row_count"]
                max_columns = max(max_columns, table["column_count"])
                sections.append(
                    {
                        "id": f"sheet_{sheet_name}",
                        "type": "sheet_summary",
                        "sheet_name": sheet_name,
                        "text": (
                            f"Sheet {sheet_name} contains {table['row_count']} rows and "
                            f"{table['column_count']} columns."
                        ),
                        "row_count": table["row_count"],
                        "column_count": table["column_count"],
                    }
                )

            summary_text = build_dataset_text_summary(path.name, tables)
            warnings = []
            if not tables:
                warnings.append("Workbook contains no readable sheets.")
            elif total_rows == 0:
                warnings.append("Workbook sheets were loaded, but no data rows were found.")

            metadata = {
                "sheet_count": len(tables),
                "sheet_names": [table["name"] for table in tables],
                "row_count": total_rows,
                "column_count": max_columns,
            }

            logger.info("Loaded XLSX content from %s", path)
            return build_success_result(
                path,
                "xlsx",
                metadata=metadata,
                text=summary_text,
                sections=sections,
                tables=tables,
                warnings=warnings,
            )
        except Exception as exc:
            logger.exception("XLSX summarization failed for %s", path)
            return build_error_result(path, "xlsx", errors=[f"XLSX summarization failed: {exc}"])

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Alias to align XLSX ingestion with extractor-style interfaces."""
        return self.load(file_path)


def load_xlsx(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for XLSX ingestion."""
    return XLSXLoader(**kwargs).load(file_path)
