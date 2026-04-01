"""PPTX ingestion utilities for CloudInsight v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from pptx import Presentation
except ImportError:  # pragma: no cover - handled at runtime
    Presentation = None  # type: ignore[assignment]


class PPTXExtractor:
    """Extract slide-wise structured content from presentation files."""

    SUPPORTED_SUFFIXES = {".pptx"}

    def __init__(self, *, max_slides: int | None = None, max_chars_per_slide: int | None = 20000) -> None:
        self.max_slides = max_slides
        self.max_chars_per_slide = max_chars_per_slide

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract PPTX content and return a standardized payload."""
        path = Path(file_path).expanduser()
        detected_type = self._detect_file_type(path)

        try:
            self._validate(path, detected_type)
            self._ensure_dependency()

            presentation = Presentation(str(path))
            total_slides = len(presentation.slides)
            slide_limit = total_slides if self.max_slides is None else min(total_slides, self.max_slides)

            sections: list[dict[str, Any]] = []
            warnings: list[str] = []
            overview_parts: list[str] = []

            for slide_index in range(slide_limit):
                slide = presentation.slides[slide_index]
                slide_text_fragments: list[str] = []

                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        for paragraph in shape.text_frame.paragraphs:
                            text = " ".join(run.text.strip() for run in paragraph.runs if run.text.strip())
                            if not text:
                                text = " ".join(paragraph.text.split())
                            if text:
                                slide_text_fragments.append(text)

                slide_text = "\n".join(slide_text_fragments).strip()
                if self.max_chars_per_slide and len(slide_text) > self.max_chars_per_slide:
                    slide_text = slide_text[: self.max_chars_per_slide].rstrip()
                    warnings.append(
                        f"Slide {slide_index} exceeded {self.max_chars_per_slide} characters and was truncated."
                    )

                sections.append(
                    {
                        "id": f"slide_{slide_index + 1}",
                        "type": "slide",
                        "slide_number": slide_index + 1,
                        "title": self._extract_title(slide),
                        "text": slide_text,
                        "char_count": len(slide_text),
                    }
                )
                if slide_text:
                    overview_parts.append(f"Slide {slide_index + 1}: {slide_text}")

            if total_slides > slide_limit:
                warnings.append(
                    f"Processed {slide_limit} of {total_slides} slides. Increase max_slides to ingest the full presentation."
                )

            metadata = {
                "file_size_bytes": path.stat().st_size,
                "slide_count": total_slides,
                "slides_processed": slide_limit,
                "truncated": total_slides > slide_limit or bool(warnings),
            }
            content = {
                "text": "\n\n".join(overview_parts),
                "sections": sections,
                "tables": [],
            }
            return self._success_response(path, detected_type, metadata, content, warnings)
        except Exception as exc:  # pragma: no cover - exercised through integration
            return self._error_response(path, detected_type, exc)

    def _extract_title(self, slide: Any) -> str:
        if getattr(slide.shapes, "title", None) is not None:
            title_text = getattr(slide.shapes.title, "text", "") or ""
            return " ".join(title_text.split())
        return ""

    def _detect_file_type(self, path: Path) -> str:
        return path.suffix.lower().lstrip(".") or "unknown"

    def _validate(self, path: Path, detected_type: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PPTX file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file path, received: {path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type for PPTX extractor: {detected_type}")

    def _ensure_dependency(self) -> None:
        if Presentation is None:
            raise ImportError("PPTX extraction requires 'python-pptx' to be installed.")

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
            "file_type": "pptx",
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
            "file_type": "pptx",
            "detected_type": detected_type,
            "metadata": {},
            "content": {"text": "", "sections": [], "tables": []},
            "warnings": [],
            "errors": [str(exc)],
        }


def extract_pptx(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for PPTX extraction."""
    return PPTXExtractor(**kwargs).extract(file_path)
