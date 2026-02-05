import chromadb
from pypdf import PdfReader

PDF_PATH = "vedic_astro_textbook.pdf"
DB_DIR = "./chroma_jyotish"
COLLECTION = "jyotish_book"

def check_pdf_text():
    print("=== Checking PDF text directly ===")
    try:
        pdf = PdfReader(PDF_PATH)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if "mrig" in text.lower():
                print(f"\nPage {page_num}:")
                # Find and show context around mrig
                lines = text.split('\n')
                for line_num, line in enumerate(lines):
                    if "mrig" in line.lower():
                        # Show this line and surrounding lines
                        start = max(0, line_num - 1)
                        end = min(len(lines), line_num + 2)
                        for i in range(start, end):
                            marker = ">>> " if i == line_num else "    "
                            print(f"{marker}{lines[i]}")
                break
    except Exception as e:
        print(f"Error reading PDF: {e}")

def check_chroma_text():
    print("\n=== Checking ChromaDB text ===")
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_or_create_collection(COLLECTION)
        
        # Get all documents and search for mrig
        all_docs = collection.get(include=["documents", "metadatas"])
        
        found_count = 0
        for i, (doc, meta) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            if "mrig" in doc.lower():
                found_count += 1
                page = meta.get('page', '?')
                print(f"\nPage {page} (doc {i}):")
                # Show snippet
                lines = doc.split('\n')
                for line_num, line in enumerate(lines):
                    if "mrig" in line.lower():
                        marker = ">>> "
                        print(f"{marker}{line}")
                        # Show surrounding context
                        start = max(0, line_num - 1)
                        end = min(len(lines), line_num + 2)
                        for j in range(start, end):
                            if j != line_num:
                                print(f"    {lines[j]}")
                        break
                if found_count >= 5:  # Limit output
                    break
        
        print(f"\nTotal documents with 'mrig': {found_count}")
        
    except Exception as e:
        print(f"Error reading ChromaDB: {e}")

if __name__ == "__main__":
    check_pdf_text()
    check_chroma_text()
