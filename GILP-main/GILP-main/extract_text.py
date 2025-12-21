
import pypdf
import os

pdf_files = [
    "input_1.pdf",
    "input_2.pdf"
]

for pdf_path in pdf_files:
    print(f"--- Processing {os.path.basename(pdf_path)} ---")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text[:5000]) # Print first 5000 chars to avoid huge output, hopefully enough to grasp the concept
        if len(text) > 5000:
            print("\n...[TRUNCATED]...")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    print("\n" + "="*50 + "\n")
