# backend/app/engine/parser.py
import urllib.request
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

class IngestionEngine:
    def __init__(self):
        """Initializes Docling's layout-parsing models and semantic chunker."""
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def download_sample(self, url: str, output_path: str):
        """Downloads a PDF from a given URL."""
        print(f"[*] Downloading sample paper from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"[*] Saved to {output_path}")

    def parse_and_chunk(self, file_path: str):
        """
        Parses the PDF retaining tables/LaTeX, then applies semantic chunking.
        """
        print(f"[*] Parsing document: {file_path}")
        print("[*] (Note: The first run takes longer as it initializes PyTorch models)")
        
        # 1. Layout-Aware Parsing
        result = self.converter.convert(file_path)
        document = result.document
        
        # 2. Extract Full Markdown (for visual verification)
        markdown_content = document.export_to_markdown()
        
        # 3. Hierarchical Chunking (keeps tables and sections structurally intact)
        chunks = list(self.chunker.chunk(document))
        
        return markdown_content, chunks

if __name__ == "__main__":
    # Setup robust absolute paths based on this file's location
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = base_dir / "data" / "sample_papers"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = data_dir / "attention_is_all_you_need.pdf"
    md_path = data_dir / "attention_parsed.md"
    
    engine = IngestionEngine()
    
    # Download the famous Transformer paper if we don't have it yet
    if not pdf_path.exists():
        engine.download_sample("https://arxiv.org/pdf/1706.03762.pdf", str(pdf_path))
        
    # Execute the parsing and chunking pipeline
    md_content, chunks = engine.parse_and_chunk(str(pdf_path))
    
    # Save the Markdown output so we can manually inspect the layout accuracy
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n[+] Success! Document parsed into {len(chunks)} hierarchical chunks.")
    print(f"[+] Markdown extracted and saved to: {md_path}")
    print("\n[NEXT STEP] -> Open data/sample_papers/attention_parsed.md in VS Code to verify the table extraction.")