
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
        print(f"Pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            print(f"--- Page {i+1} ---")
            text = page.extract_text()
            print(text)
            if i > 2: # Check first 3 pages only to save space
                print("...Stopping after 3 pages...")
                break
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    print("\n" + "="*50 + "\n")
