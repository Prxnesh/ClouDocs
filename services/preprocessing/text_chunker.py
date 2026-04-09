import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles chunking of large text documents into smaller, semantically meaningful pieces
    suitable for vector embeddings and LLM context windows.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """
        Splits text into chunks using a sliding window approach.
        Tries to avoid splitting words in half by scanning for spaces.
        
        Args:
            text (str): The large body of text to chunk.
            
        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # If we're not at the end of the text, try to find a nice breaking point
            if end < text_length:
                # Look backwards for a newline or space within the last 100 chars of the chunk
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                # Prefer newlines for cleaner semantic breaks, fallback to spaces
                if last_newline != -1 and (end - last_newline) < 100:
                    end = last_newline + 1
                elif last_space != -1 and (end - last_space) < 50:
                    end = last_space + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start pointer forward, accounting for overlap
            # Ensure we actually advance to avoid infinite loops if overlap >= chunk_size
            advance = max(1, end - start - self.overlap)
            start += advance

        logger.info(f"Chunked text of length {text_length} into {len(chunks)} chunks.")
        return chunks
