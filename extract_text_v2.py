
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
