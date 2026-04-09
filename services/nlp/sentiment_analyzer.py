from collections import Counter
import logging
from textblob import TextBlob
from typing import Dict, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a given text block using TextBlob.
    
    Args:
        text (str): The body of text to analyze.
        
    Returns:
        Dict[str, Any]: A dictionary containing ui-compatible sentiment results.
    """
    if not text or not str(text).strip():
        logger.warning("No text provided for sentiment analysis.")
        return {
            "polarity_score": 0.0, 
            "confidence": 0.0, 
            "label": "Neutral",
            "guidance": "Provide more text for analysis.",
            "positive_hits": {},
            "negative_hits": {}
        }

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        label = "Positive"
        guidance = "The tone leans positive and confident."
    elif polarity < -0.05:
        label = "Negative"
        guidance = "The tone reveals some critical or negative elements."
    else:
        label = "Neutral"
        guidance = "The tone is highly objective and balanced."

    # Extract mock hits to satisfy the UI dashboard's dictionary requirements
    positive_bag = {"good", "great", "excellent", "fast", "brilliantly", "success", "successful", "insight", "intelligence"}
    negative_bag = {"bad", "terrible", "slow", "error", "fail", "failed", "crash", "issue", "bug"}
    
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    pos_hits = Counter([w for w in words if w in positive_bag])
    neg_hits = Counter([w for w in words if w in negative_bag])

    logger.info(f"Sentiment analysis completed: {label} (polarity: {polarity:.2f})")
    
    return {
        "polarity_score": round(polarity, 4),
        "confidence": round(subjectivity if subjectivity > 0 else 0.8, 4),
        "label": label,
        "guidance": guidance,
        "positive_hits": dict(pos_hits),
        "negative_hits": dict(neg_hits)
    }
