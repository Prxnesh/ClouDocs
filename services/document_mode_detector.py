"""Smart document mode detection for CloudInsight v2.

This module provides a lightweight document-type detector that combines
rule-based signals with optional upstream classification hints. It is designed
to sit after ingestion and before deeper NLP analysis so the pipeline can adapt
to the kind of document being processed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


SECTION_BREAK_PATTERN = re.compile(r"\n\s*\n+")
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d{2})?|\b(?:total|amount due|invoice total|balance due)[:\s$]*\d[\d,]*(?:\.\d{2})?\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModeProfile:
    """Configuration for a supported document mode."""

    mode: str
    display_name: str
    keywords: tuple[str, ...]
    extraction_targets: tuple[str, ...]
    recommended_analyses: tuple[str, ...]


@dataclass
class ModeDetection:
    """Serializable result for document mode detection."""

    mode: str
    display_name: str
    confidence: float
    matched_signals: list[str]
    scores: dict[str, float]
    extraction_targets: list[str]
    recommended_analyses: list[str]
    extracted_fields: dict[str, Any]


MODE_PROFILES: dict[str, ModeProfile] = {
    "resume": ModeProfile(
        mode="resume",
        display_name="Resume",
        keywords=(
            "experience",
            "education",
            "skills",
            "projects",
            "certifications",
            "curriculum vitae",
            "employment history",
        ),
        extraction_targets=("skills", "education", "experience"),
        recommended_analyses=("keyword_matching", "candidate_scoring", "entity_extraction"),
    ),
    "invoice": ModeProfile(
        mode="invoice",
        display_name="Invoice",
        keywords=(
            "invoice",
            "invoice number",
            "bill to",
            "amount due",
            "balance due",
            "vendor",
            "subtotal",
            "tax",
            "payment terms",
        ),
        extraction_targets=("amount", "date", "vendor"),
        recommended_analyses=("amount_extraction", "date_extraction", "vendor_detection"),
    ),
    "research_paper": ModeProfile(
        mode="research_paper",
        display_name="Research Paper",
        keywords=(
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "keywords",
        ),
        extraction_targets=("abstract", "conclusion"),
        recommended_analyses=("section_summarization", "entity_extraction", "topic_detection"),
    ),
    "general_document": ModeProfile(
        mode="general_document",
        display_name="General Document",
        keywords=(),
        extraction_targets=("summary", "keywords", "entities"),
        recommended_analyses=("summarization", "keyword_extraction", "classification"),
    ),
}


CLASSIFICATION_TO_MODE = {
    "resume": "resume",
    "cv": "resume",
    "invoice": "invoice",
    "bill": "invoice",
    "research_paper": "research_paper",
    "paper": "research_paper",
    "academic_paper": "research_paper",
}


class DocumentModeDetector:
    """Detect document mode and return mode-specific analysis instructions."""

    def detect(
        self,
        text: str,
        *,
        file_name: str | None = None,
        classification_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect the document mode from text, file name, and optional hints."""
        normalized_text = self._normalize_text(text)
        scores = self._score_modes(normalized_text, file_name=file_name, classification_label=classification_label)
        mode = max(scores, key=scores.get)
        profile = MODE_PROFILES[mode]
        extracted_fields = self._extract_fields(mode, text, metadata=metadata or {})

        total_score = sum(scores.values()) or 1.0
        confidence = round(scores[mode] / total_score, 3)

        detection = ModeDetection(
            mode=profile.mode,
            display_name=profile.display_name,
            confidence=confidence,
            matched_signals=self._matched_signals(profile.mode, normalized_text, file_name, classification_label),
            scores={name: round(value, 3) for name, value in scores.items()},
            extraction_targets=list(profile.extraction_targets),
            recommended_analyses=list(profile.recommended_analyses),
            extracted_fields=extracted_fields,
        )
        return asdict(detection)

    def _score_modes(
        self,
        normalized_text: str,
        *,
        file_name: str | None,
        classification_label: str | None,
    ) -> dict[str, float]:
        """Score each supported mode using rule signals and optional classifier hints."""
        lowered_name = (file_name or "").lower()
        scores = {mode: 0.2 for mode in MODE_PROFILES}

        for mode, profile in MODE_PROFILES.items():
            for keyword in profile.keywords:
                occurrences = normalized_text.count(keyword.lower())
                if occurrences:
                    scores[mode] += min(occurrences, 4) * 1.2

            if lowered_name:
                for keyword in profile.keywords[:4]:
                    if keyword.lower().replace(" ", "_") in lowered_name or keyword.lower() in lowered_name:
                        scores[mode] += 1.4

        self._apply_structural_boosts(scores, normalized_text)
        self._apply_classification_hint(scores, classification_label)
        return scores

    def _apply_structural_boosts(self, scores: dict[str, float], normalized_text: str) -> None:
        """Boost modes using document structure that simple keywords might miss."""
        if DATE_PATTERN.search(normalized_text) and CURRENCY_PATTERN.search(normalized_text):
            scores["invoice"] += 2.8

        if self._contains_any(normalized_text, ("skills", "education", "experience")):
            scores["resume"] += 2.5

        if self._contains_any(normalized_text, ("abstract", "conclusion", "references")):
            scores["research_paper"] += 3.0

    def _apply_classification_hint(self, scores: dict[str, float], classification_label: str | None) -> None:
        """Blend an upstream classification hint into the final mode score."""
        if not classification_label:
            return

        normalized_label = classification_label.strip().lower().replace(" ", "_")
        mapped_mode = CLASSIFICATION_TO_MODE.get(normalized_label)
        if mapped_mode:
            scores[mapped_mode] += 4.0

    def _matched_signals(
        self,
        mode: str,
        normalized_text: str,
        file_name: str | None,
        classification_label: str | None,
    ) -> list[str]:
        """Return the signals that contributed most to the detected mode."""
        profile = MODE_PROFILES[mode]
        signals: list[str] = []

        if classification_label and CLASSIFICATION_TO_MODE.get(classification_label.strip().lower().replace(" ", "_")) == mode:
            signals.append(f"classification_hint:{classification_label}")

        for keyword in profile.keywords:
            if keyword.lower() in normalized_text:
                signals.append(f"keyword:{keyword}")
            if len(signals) >= 5:
                break

        lowered_name = (file_name or "").lower()
        if lowered_name:
            for keyword in profile.keywords:
                if keyword.lower() in lowered_name or keyword.lower().replace(" ", "_") in lowered_name:
                    signals.append(f"filename:{keyword}")
                    break

        if mode == "invoice" and DATE_PATTERN.search(normalized_text) and CURRENCY_PATTERN.search(normalized_text):
            signals.append("structure:date+currency")
        if mode == "resume" and self._contains_any(normalized_text, ("skills", "education", "experience")):
            signals.append("structure:resume_sections")
        if mode == "research_paper" and self._contains_any(normalized_text, ("abstract", "conclusion", "references")):
            signals.append("structure:paper_sections")

        return signals[:6]

    def _extract_fields(self, mode: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract lightweight structured fields for the detected mode."""
        if mode == "resume":
            return self._extract_resume_fields(text)
        if mode == "invoice":
            return self._extract_invoice_fields(text, metadata=metadata)
        if mode == "research_paper":
            return self._extract_research_fields(text)
        return {}

    def _extract_resume_fields(self, text: str) -> dict[str, Any]:
        """Extract lightweight resume sections and likely skills."""
        skills_block = self._extract_section_block(text, ("skills", "technical skills", "core competencies"))
        education_block = self._extract_section_block(text, ("education", "academic background"))
        experience_block = self._extract_section_block(text, ("experience", "work experience", "employment history"))

        skill_dictionary = {
            "python",
            "sql",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "nlp",
            "streamlit",
            "aws",
            "gcp",
            "docker",
            "kubernetes",
            "fastapi",
            "pandas",
            "numpy",
            "power bi",
            "tableau",
            "excel",
            "java",
            "javascript",
            "typescript",
        }
        detected_skills = sorted(skill for skill in skill_dictionary if skill in text.lower())

        return {
            "skills": detected_skills or self._block_to_items(skills_block),
            "education": self._block_to_items(education_block),
            "experience": self._block_to_items(experience_block),
        }

    def _extract_invoice_fields(self, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract likely invoice date, amount, and vendor."""
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        vendor = metadata.get("source_vendor") or (non_empty_lines[0] if non_empty_lines else "")

        amount_match = CURRENCY_PATTERN.search(text)
        date_match = DATE_PATTERN.search(text)

        return {
            "vendor": vendor,
            "date": date_match.group(0) if date_match else "",
            "amount": amount_match.group(0) if amount_match else "",
        }

    def _extract_research_fields(self, text: str) -> dict[str, Any]:
        """Extract abstract and conclusion blocks from a research paper."""
        abstract = self._extract_section_block(text, ("abstract",))
        conclusion = self._extract_section_block(text, ("conclusion", "conclusions"))
        return {
            "abstract": abstract,
            "conclusion": conclusion,
        }

    def _extract_section_block(self, text: str, headings: tuple[str, ...]) -> str:
        """Extract text from a headed section using simple section boundaries."""
        sections = SECTION_BREAK_PATTERN.split(text)
        for index, section in enumerate(sections):
            header = section.strip().splitlines()[0].strip(" :").lower() if section.strip() else ""
            if any(header.startswith(heading.lower()) for heading in headings):
                return sections[index].strip()

        text_lower = text.lower()
        for heading in headings:
            anchor = text_lower.find(heading.lower())
            if anchor != -1:
                snippet = text[anchor : anchor + 1200]
                return snippet.strip()
        return ""

    def _block_to_items(self, block: str, limit: int = 6) -> list[str]:
        """Convert a section block into short list items."""
        if not block:
            return []
        lines = [line.strip(" -•\t") for line in block.splitlines() if line.strip()]
        if len(lines) <= 1:
            segments = [segment.strip(" -•\t") for segment in re.split(r"[;,]", block) if segment.strip()]
            return segments[:limit]
        return lines[1 : limit + 1]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for scoring."""
        lowered = text.lower()
        return re.sub(r"\s+", " ", lowered)

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        """Check whether any keyword appears in the normalized text."""
        return any(keyword in text for keyword in keywords)


def detect_document_mode(
    text: str,
    *,
    file_name: str | None = None,
    classification_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for smart document mode detection."""
    detector = DocumentModeDetector()
    return detector.detect(
        text,
        file_name=file_name,
        classification_label=classification_label,
        metadata=metadata,
    )
