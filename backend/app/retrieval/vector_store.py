# backend/app/retrieval/vector_store.py
import os
import sys
import uuid
from pathlib import Path

# 1. FORCE LOAD FASTEMBED FIRST (Prevents ONNX DLL collisions with Docling)
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Ensure backend is in path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.engine.parser import IngestionEngine

class HybridVectorStore:
    def __init__(self, collection_name: str = "ai_research_papers"):
        self.collection_name = collection_name
        
        # Robust path resolution: prioritize ENV variable, fallback to local relative path
        env_path = os.getenv("QDRANT_STORAGE_PATH")
        if env_path:
            self.storage_path = Path(env_path)
        else:
            # Local fallback: backend/app/retrieval -> root / data / qdrant_db
            self.storage_path = Path(__file__).resolve().parents[3] / "data" / "qdrant_db"
            
        print(f"[*] Initializing Qdrant Local Storage at {self.storage_path}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # --- MLOps Decoupling: We manage Compute (Embeddings) separately from Storage ---
        print("[*] Initializing FastEmbed Compute Models...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        self.client = QdrantClient(path=str(self.storage_path))

        # Initialize Collection Schema strictly if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            print("[*] Creating Hybrid Vector Schema in Qdrant...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,  # BGE-Small exact dimension
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def index_document(self, file_path: str, source_file: str | None = None):
        source_file = source_file or os.path.basename(file_path)
        if source_file in self.get_indexed_files():
            return 0

        engine = IngestionEngine()
        _, doc_chunks = engine.parse_and_chunk(file_path)
        print(f"[*] Extracted {len(doc_chunks)} chunks. Generating explicit embeddings...")

        texts = []
        payloads = []
        for chunk in doc_chunks:
            text = chunk.text
            if not text.strip():
                continue
            texts.append(text)
            heading = chunk.meta.headings[0] if chunk.meta.headings else "General"
            page = self._get_page_number(chunk)
            
            # Save the text in payload so we can retrieve it
            payload = {
                "source_file": source_file,
                "heading": heading,
                "text": text,
            }
            if page is not None:
                payload["page"] = page
            payloads.append(payload)

        if not texts:
            print("[!] No text chunks were extracted; nothing was indexed.")
            return 0

        # Generate vectors mathematically
        dense_vectors = list(self.dense_model.embed(texts))
        sparse_vectors = list(self.sparse_model.embed(texts))

        print("[*] Uploading vector payloads to Qdrant...")
        points = []
        for i in range(len(texts)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload=payloads[i],
                    vector={
                        "dense": dense_vectors[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[i].indices.tolist(),
                            values=sparse_vectors[i].values.tolist()
                        )
                    }
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[+] Successfully indexed {len(points)} chunks.")
        return len(points)

    def get_indexed_files(self) -> set[str]:
        """Return source filenames currently represented in the collection."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=["source_file"],
            with_vectors=False,
        )
        return {
            source_file
            for point in points
            if (source_file := (point.payload or {}).get("source_file"))
        }

    @staticmethod
    def _get_page_number(chunk) -> int | None:
        """Read the first provenance page when Docling exposes it."""
        doc_items = getattr(getattr(chunk, "meta", None), "doc_items", None)
        if not doc_items:
            return None

        provenance = getattr(doc_items[0], "prov", None)
        if not provenance:
            return None

        page_number = getattr(provenance[0], "page_no", None)
        return page_number if isinstance(page_number, int) and page_number >= 1 else None

    def hybrid_search(self, query: str, limit: int = 5):
        print(f"\n[*] Executing Hybrid RRF Search for: '{query}'")
        
        # Embed the query
        query_dense = list(self.dense_model.embed([query]))[0].tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]

        # Execute Reciprocal Rank Fusion (RRF) combining both vector spaces
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=limit
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        return results.points


if __name__ == "__main__":
    store = HybridVectorStore()
    pdf_path = store.base_dir / "data" / "sample_papers" / "attention_is_all_you_need.pdf"
    
    # 1. Index the paper
    store.index_document(str(pdf_path))
    
    # 2. Test the Hybrid Retrieval
    test_query = "What optimizer, learning rate schedule, and warmup steps did they use for training?"
    search_results = store.hybrid_search(test_query, limit=3)
    
    print("\n[--- Top Retrieval Results ---]")
    for i, res in enumerate(search_results, 1):
        heading = res.payload.get("heading", "N/A")
        document_text = res.payload.get("text", "")
        print(f"\nResult {i} (Score: {res.score:.4f}) | Section: {heading}")
        print(f"Snippet: {document_text[:300]}...")