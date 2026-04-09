import logging
from collections import defaultdict
import string
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure nltk packages are downloaded implicitly if possible
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractiveSummarizer:
    """
    Provides lightweight, local extractive text summarization by ranking
    sentences based on word frequency.
    """
    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences

    def summarize(self, text: str) -> str:
        """
        Summarizes the text by extracting the most salient sentences.
        """
        if not text or len(text.split()) < 10:
            return text

        try:
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))

            sentences = sent_tokenize(text)
            if len(sentences) <= self.num_sentences:
                return text

            # Calculate word frequencies
            words = word_tokenize(text.lower())
            freq_table = defaultdict(int)
            for word in words:
                if word not in stop_words and word not in string.punctuation:
                    freq_table[word] += 1

            # Score sentences based on frequency of salient words
            sentence_scores = defaultdict(int)
            for i, sentence in enumerate(sentences):
                sentence_words = word_tokenize(sentence.lower())
                for word in sentence_words:
                    if word in freq_table:
                        sentence_scores[i] += freq_table[word]

            # Get top N sentence indices
            top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:self.num_sentences]
            top_sentence_indices.sort() # sort sequentially

            summary = ' '.join([sentences[i] for i in top_sentence_indices])
            logger.info("Successfully produced an extractive summary.")
            return summary

        except Exception as e:
            logger.error(f"Error summarising text: {e}")
            # Fallback to simple truncation
            return text[:500] + "..." if len(text) > 500 else text
