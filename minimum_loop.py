from sentence_transformers import SentenceTransformer
import chromadb

import pprint

client = chromadb.Client()
#model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

collection = client.create_collection(
    name="min_loop"
)

query = "What is the capital of California?"

docs = [
    "California is a member of the United States of America.",
    "The capital of Arizona is Phoenix",
    "The largest city in California is Los Angeles",
    "The capital of California is Sacremento"
]

collection.add(
    ids=[str(i) for i in range(len(docs))],
    documents=docs
)

results = collection.query(
    query_texts=query,
    n_results=3
)


pprint.pprint(results)

print(results["documents"])
