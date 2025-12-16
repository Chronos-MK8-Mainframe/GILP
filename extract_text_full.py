
from pdfminer.high_level import extract_text
import os

pdf_files = [
    r"c:\Users\rupa9\Videos\GILP\GILP_solutions.pdf",
    r"c:\Users\rupa9\Videos\GILP\ACFrOgDydjcahXK-WT_66RSYi1_0VkyH1iiZGWSaeHMZMLStQEq20zR-g3ixfGgmSbY1XYK0V-gSxKxBgEzmiIwUbhGbxXkXifjMNcVrHuX3bWlwi9MiQQCVe2qjUigwscZdO0IyNLxoN0OkpP9_.pdf"
]

output_file = r"c:\Users\rupa9\Videos\GILP\full_text.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for pdf_path in pdf_files:
        f.write(f"\n\n{'='*50}\nPROCESSING {os.path.basename(pdf_path)}\n{'='*50}\n\n")
        try:
            text = extract_text(pdf_path)
            f.write(text)
        except Exception as e:
            f.write(f"\nError reading {pdf_path}: {e}\n")

print(f"Text extracted to {output_file}")
