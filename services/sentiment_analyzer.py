"""Lightweight sentiment analysis utilities for CloudInsight v2."""

from __future__ import annotations

from collections import Counter
from typing import Any


POSITIVE_WORDS = {
    "accelerated",
    "accomplished",
    "achieved",
    "beneficial",
    "clear",
    "confident",
    "delight",
    "effective",
    "efficient",
    "excellent",
    "excited",
    "favorable",
    "good",
    "great",
    "improved",
    "improving",
    "insightful",
    "optimistic",
    "outstanding",
    "positive",
    "reliable",
    "resilient",
    "robust",
    "smooth",
    "strong",
    "successful",
    "valuable",
    "win",
}

NEGATIVE_WORDS = {
    "angry",
    "bad",
    "blocked",
    "confused",
    "concern",
    "concerning",
    "critical",
    "delayed",
    "difficult",
    "disappointing",
    "error",
    "fail",
    "failing",
    "frustrated",
    "hard",
    "issue",
    "loss",
    "negative",
    "poor",
    "problem",
    "risky",
    "severe",
    "slow",
    "stress",
    "stuck",
    "uncertain",
    "weak",
    "worse",
}


class SentimentAnalyzer:
    """Analyze sentiment using a lightweight lexicon-based approach."""

    def analyze(self, text: str) -> dict[str, Any]:
        """Return sentiment label, score, matched cues, and writing guidance."""
        tokens = self._tokenize(text)
        counts = Counter(tokens)

        positive_hits = {word: counts[word] for word in POSITIVE_WORDS if counts[word]}
        negative_hits = {word: counts[word] for word in NEGATIVE_WORDS if counts[word]}

        positive_score = sum(positive_hits.values())
        negative_score = sum(negative_hits.values())
        total_hits = positive_score + negative_score

        if total_hits == 0:
            polarity_score = 0.0
        else:
            polarity_score = (positive_score - negative_score) / total_hits

        label = self._label_from_score(polarity_score)
        confidence = round(min(1.0, total_hits / 8), 3)

        return {
            "label": label,
            "polarity_score": round(polarity_score, 3),
            "confidence": confidence,
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "guidance": self._guidance_from_label(label),
        }

    def _tokenize(self, text: str) -> list[str]:
        """Convert raw text into lowercase word tokens."""
        cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
        return [token for token in cleaned.split() if token]

    def _label_from_score(self, score: float) -> str:
        """Map a polarity score to a human-readable sentiment label."""
        if score >= 0.25:
            return "Positive"
        if score <= -0.25:
            return "Negative"
        return "Neutral"

    def _guidance_from_label(self, label: str) -> str:
        """Return concise editorial guidance based on detected sentiment."""
        if label == "Positive":
            return "The draft reads upbeat and confident. Keep the tone if you want a persuasive or encouraging message."
        if label == "Negative":
            return "The draft carries noticeable friction or concern. Consider softening harsh phrasing if the goal is alignment."
        return "The draft reads balanced and informational. Add stronger emotional language only if you want more urgency or warmth."


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Convenience wrapper for sentiment analysis."""
    return SentimentAnalyzer().analyze(text)
