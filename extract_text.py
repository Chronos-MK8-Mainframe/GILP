
import pypdf
import os

pdf_files = [
    r"c:\Users\rupa9\Videos\GILP\ACFrOgDydjcahXK-WT_66RSYi1_0VkyH1iiZGWSaeHMZMLStQEq20zR-g3ixfGgmSbY1XYK0V-gSxKxBgEzmiIwUbhGbxXkXifjMNcVrHuX3bWlwi9MiQQCVe2qjUigwscZdO0IyNLxoN0OkpP9_.pdf",
    r"c:\Users\rupa9\Videos\GILP\GILP_solutions.pdf"
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
