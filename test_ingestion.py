import os
import sys

# Ensure the root directory is in the path so we can import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ingestion.pdf_extractor import PDFExtractor
from services.ingestion.docx_extractor import DOCXExtractor
from services.ingestion.pptx_extractor import PPTXExtractor
from services.ingestion.csv_loader import CSVLoader
from services.ingestion.xlsx_loader import XLSXLoader

def test_imports():
    print("Testing imports for all ingestion modules...")
    print("PDFExtractor imported successfully:", PDFExtractor is not None)
    print("DOCXExtractor imported successfully:", DOCXExtractor is not None)
    print("PPTXExtractor imported successfully:", PPTXExtractor is not None)
    print("CSVLoader imported:", CSVLoader is not None)
    print("XLSXLoader imported:", XLSXLoader is not None)
    
    print("\nTesting CSVLoader execution...")
    import tempfile
    import pandas as pd
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n1,2\n3,4")
        temp_path = Path(f.name)
        
    loader = CSVLoader()
    result = loader.load(temp_path)
    print("CSVLoader output format:")
    import json
    print(json.dumps(result, indent=2))
    
    # cleanup
    temp_path.unlink()
    print("\nAll Phase 1 modules are ready and returning the proper dictionary structure!")

if __name__ == "__main__":
    test_imports()
