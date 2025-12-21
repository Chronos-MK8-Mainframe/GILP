
from pdfminer.high_level import extract_text
import os

pdf_files = [
    "input_1.pdf",
    "input_2.pdf"
]

output_file = "full_text.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for pdf_path in pdf_files:
        f.write(f"\n\n{'='*50}\nPROCESSING {os.path.basename(pdf_path)}\n{'='*50}\n\n")
        try:
            text = extract_text(pdf_path)
            f.write(text)
        except Exception as e:
            f.write(f"\nError reading {pdf_path}: {e}\n")

print(f"Text extracted to {output_file}")
