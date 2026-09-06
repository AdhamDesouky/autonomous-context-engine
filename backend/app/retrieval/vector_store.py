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
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.db_path = self.base_dir / "data" / "qdrant_db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = "ai_research_papers"
        
        print(f"[*] Initializing Qdrant Local Storage at {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))
        
        # --- MLOps Decoupling: We manage Compute (Embeddings) separately from Storage ---
        print("[*] Initializing FastEmbed Compute Models...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

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

    def index_document(self, file_path: str):
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
            
            # Save the text in payload so we can retrieve it
            payloads.append({"source_file": os.path.basename(file_path), "heading": heading, "text": text})

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