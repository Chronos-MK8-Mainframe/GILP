
from pdfminer.high_level import extract_text
import os

pdf_files = [
    "input_1.pdf",
    "input_2.pdf"
]

for pdf_path in pdf_files:
    print(f"--- Processing {os.path.basename(pdf_path)} ---")
    try:
        text = extract_text(pdf_path)
        print(text[:5000]) 
        if len(text) > 5000:
            print("\n...[TRUNCATED]...")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    print("\n" + "="*50 + "\n")
