import os

import chromadb
from anthropic import AnthropicFoundry
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from chunker import load_docs, split_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

SYSTEM_PROMPT = """You answer questions about Arasaka company policy.
Use only the context provided in the user's message.
If the context does not contain the answer, say you don't know."""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""


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


def build_prompt(question, hits, show_sources=True):
    if show_sources:
        context = "\n\n".join(
            f"[{hit['metadata'].get('source', '?')}] {hit['text']}" for hit in hits
        )
    else:
        context = "\n\n".join(hit["text"] for hit in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer(question, collection, llm, k=3, show_sources=True):
    """Retrieve context, ask llm (a prompt -> str callable), and report sources."""
    hits = retrieve(collection, question, k=k)
    sources = list(dict.fromkeys(hit["metadata"]["source"] for hit in hits))
    prompt = build_prompt(question, hits, show_sources=show_sources)
    return {"answer": llm(prompt), "sources": sources}


def make_foundry_llm():
    """Return a prompt -> str callable backed by a Claude deployment on Microsoft Foundry."""
    load_dotenv()
    llm_client = AnthropicFoundry(
        api_key=os.environ["FOUNDRY_API_KEY"],
        base_url=os.environ["FOUNDRY_ENDPOINT"],
    )
    deployment = os.environ["FOUNDRY_DEPLOYMENT"]

    def llm(prompt):
        message = llm_client.messages.create(
            model=deployment,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        if message.stop_reason == "refusal":
            return "The model declined to answer this question."
        return "".join(block.text for block in message.content if block.type == "text")

    return llm


if __name__ == "__main__":
    collection = build_index(split_chunks(load_docs("./policies/")), "Arasaka_docs")
    query = "How many sick days do employees get?"
    result = answer(query, collection, llm=make_foundry_llm())
    print("---------------TEST QUERY---------------")
    print(f"Query: {query}")
    print(result["answer"])
    print("Sources:", ", ".join(result["sources"]))
