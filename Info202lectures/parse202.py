import pdfplumber
from pathlib import Path

# Get all PDF files in the current directory
pdf_folder = Path(__file__).parent
pdf_files = sorted(pdf_folder.glob("*.pdf"))

all_text = []

# Process each PDF file
for pdf_file in pdf_files:
    print(f"Processing {pdf_file.name}...")

    # Add a header for each PDF
    all_text.append(f"\n{'='*80}\n")
    all_text.append(f"FILE: {pdf_file.name}\n")
    all_text.append(f"{'='*80}\n\n")

    # Extract text from each page
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(f"--- Page {page_num} ---\n")
                all_text.append(text)
                all_text.append("\n\n")

# Join all text and save to output file
full_text = "".join(all_text)
output_file = pdf_folder / "all_lectures_parsed.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\nAll PDFs parsed successfully!")
print(f"Output saved to: {output_file}")
