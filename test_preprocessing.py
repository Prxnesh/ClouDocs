import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.preprocessing.text_chunker import TextChunker
from services.preprocessing.data_cleaner import DataCleaner

def test_preprocessing():
    print("Testing Phase 2: Preprocessing Modules...\n")

    # 1. Test Text Chunker
    print("--- Testing TextChunker ---")
    chunker = TextChunker(chunk_size=50, overlap=10)
    sample_text = (
        "CloudInsight is a modular AI knowledge intelligence workspace. "
        "It supports processing complex documents quickly and locally. "
        "This is an example text to demonstrate the chunking mechanism."
    )
    chunks = chunker.chunk(sample_text)
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1} ({len(c)} chars): '{c}'")
        
    # 2. Test Data Cleaner
    print("\n--- Testing DataCleaner ---")
    import numpy as np
    raw_data = pd.DataFrame({
        "ID": [1, 2, 2, 4, 5],            # Intentionally duplicate rows for ID 2 (if rest match)
        "Name": ["Alice", "Bob", "Bob", None, "Eva"],
        "Score": [88.5, 92.0, 92.0, np.nan, 75.0]
    })
    # Add a completely empty column to be dropped
    raw_data["EmptyCol"] = np.nan

    print("Raw Data:")
    print(raw_data)
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(raw_data)
    
    print("\nCleaned Data:")
    print(cleaned_data)

if __name__ == "__main__":
    test_preprocessing()
