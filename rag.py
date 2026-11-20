from sentence_transformers import SentenceTransformer
import chromadb
from chunker import load_docs, split_chunks

import pprint

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.create_collection(
    "Arasaka_docs",
    configuration={"hnsw":{"space": "cosine"}}
    )
docs = load_docs("./policies/")

chunks = split_chunks(docs)

texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

embeddings = model.encode(texts, normalize_embeddings=True)

collection.add(
    ids=[chunk.metadata["id"] for chunk in chunks],
    documents = texts,
    metadatas=metadatas,
    embeddings=embeddings,
)

query = "What is the PTO policy?"
embedded_query = model.encode([query], normalize_embeddings=True)

results = collection.query(
    query_embeddings=embedded_query,
    n_results=3
)

pprint.pprint(results)