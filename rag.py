import pprint

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import load_docs, split_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


def build_index(chunks, collection_name="policies"):
    """Embed every chunk and store it in a new chromadb collection."""
    collection = client.create_collection(
        collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
    texts = [chunk.page_content for chunk in chunks]
    collection.add(
        ids=[chunk.metadata["id"] for chunk in chunks],
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embed(texts),
    )
    return collection


def retrieve(collection, question, k=3):
    """Return the top-k chunks for question, best match first."""
    results = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{hit['metadata'].get('source', '?')}] {hit['text']}" for hit in hits
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer(question, collection, llm, k=3):
    """Retrieve context, ask llm (a prompt -> str callable), and report sources."""
    hits = retrieve(collection, question, k=k)
    sources = list(dict.fromkeys(hit["metadata"]["source"] for hit in hits))
    return {"answer": llm(build_prompt(question, hits)), "sources": sources}


if __name__ == "__main__":
    collection = build_index(split_chunks(load_docs("./policies/")), "Arasaka_docs")
    pprint.pprint(retrieve(collection, "What is the PTO policy?"))
