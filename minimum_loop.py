"""Minimal embed-store-retrieve loop (Suggested Approach steps 2-3 in rag-guideines.pdf).

Step 2: embed one short text with sentence-transformers and store it in chromadb.
Step 3: add a second, different text, run a query, and confirm the more relevant one comes back first.
"""
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("min_loop", configuration={"hnsw": {"space": "cosine"}})


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


# ---------- Step 2: one text, embedded and stored ----------
first = "The capital of California is Sacramento."
collection.add(ids=["capital"], documents=[first], embeddings=embed([first]))
print("STEP 2: embed and store one text")
print(f"  stored: {first!r}")
print(f"  collection count: {collection.count()}\n")

# ---------- Step 3: add a second text, query, check ranking ----------
second = "The largest city in California is Los Angeles."
collection.add(ids=["largest_city"], documents=[second], embeddings=embed([second]))

query = "What is the capital of California?"
results = collection.query(query_embeddings=embed([query]), n_results=2)

print("STEP 3: add a second text and query")
print(f"  stored: {second!r}")
print(f"  collection count: {collection.count()}")
print(f"  query: {query!r}\n")
print("  results (best first, lower distance = more similar):")
for rank, (doc_id, doc, distance) in enumerate(
    zip(results["ids"][0], results["documents"][0], results["distances"][0]), start=1
):
    print(f"    {rank}. [{doc_id}] distance={distance:.3f}  {doc}")

top = results["ids"][0][0]
print(f"\n  most relevant text returned first: {'YES' if top == 'capital' else 'NO'}")
