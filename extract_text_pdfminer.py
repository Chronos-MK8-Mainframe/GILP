
from pdfminer.high_level import extract_text
import os

pdf_files = [
    r"c:\Users\rupa9\Videos\GILP\GILP_solutions.pdf",
    r"c:\Users\rupa9\Videos\GILP\ACFrOgDydjcahXK-WT_66RSYi1_0VkyH1iiZGWSaeHMZMLStQEq20zR-g3ixfGgmSbY1XYK0V-gSxKxBgEzmiIwUbhGbxXkXifjMNcVrHuX3bWlwi9MiQQCVe2qjUigwscZdO0IyNLxoN0OkpP9_.pdf"
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
