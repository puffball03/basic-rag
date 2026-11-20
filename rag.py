import os
import re

import chromadb
from anthropic import AnthropicFoundry
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunker import load_docs, split_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.Client()

SYSTEM_PROMPT = """You answer questions about Arasaka company policy.
Use only the context provided in the user's message.
If the context does not contain the answer, say you don't know."""

CURRENT_ONLY = {"status": "current"}  # leave out chunks from superseded documents


def status_filter(include_superseded):
    return None if include_superseded else CURRENT_ONLY

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
        metadatas=[{"status": "current", **chunk.metadata} for chunk in chunks],
        embeddings=embed(texts),
    )
    return collection


def retrieve(collection, question, k=3, include_superseded=False):
    """Return the top-k chunks for question, best match first."""
    results = collection.query(query_embeddings=embed([question]), n_results=k, where=status_filter(include_superseded))
    return [
        {"id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it",
    "many", "much", "my", "of", "on", "or", "the", "to", "what", "when", "where",
    "which", "who", "why", "with",
}


def tokenize(text):
    """Lowercase words minus stopwords, keeping hyphenated codes like 'hr-201' as one token."""
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())) - STOPWORDS


def keyword_search(collection, query, k=3, include_superseded=False):
    """Rank chunks by the query terms they contain, rare terms counting more."""
    stored = collection.get(where=status_filter(include_superseded))
    chunk_tokens = [tokenize(text) for text in stored["documents"]]
    terms = tokenize(query)
    doc_freq = {t: sum(t in tokens for tokens in chunk_tokens) for t in terms}

    scored = []
    for chunk_id, text, metadata, tokens in zip(
        stored["ids"], stored["documents"], stored["metadatas"], chunk_tokens
    ):
        score = sum(1 / doc_freq[t] for t in terms if t in tokens)
        if score > 0:
            scored.append((score, {"id": chunk_id, "text": text, "metadata": metadata}))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored[:k]]


RRF_K = 60  # standard Reciprocal Rank Fusion constant; damps the gap between top ranks
CANDIDATES = 10  # how many hits each search contributes before merging


def hybrid_retrieve(collection, question, k=3, include_superseded=False):
    """Merge vector and keyword results with Reciprocal Rank Fusion, best first."""
    pool = min(CANDIDATES, collection.count())
    scores, hits_by_id = {}, {}
    for results in (
        retrieve(collection, question, k=pool, include_superseded=include_superseded),
        keyword_search(collection, question, k=pool, include_superseded=include_superseded),
    ):
        for rank, hit in enumerate(results, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0) + 1 / (RRF_K + rank)
            hits_by_id[hit["id"]] = hit

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [hits_by_id[chunk_id] for chunk_id in ranked[:k]]


def rerank(question, hits, top_n=3):
    """Re-score hits with a cross-encoder (question and chunk read together); keep the best top_n."""
    scores = reranker.predict([(question, hit["text"]) for hit in hits])
    scored = [{**hit, "score": float(score)} for hit, score in zip(hits, scores)]
    scored.sort(key=lambda hit: hit["score"], reverse=True)
    return scored[:top_n]


def build_prompt(question, hits, show_sources=True):
    if show_sources:
        context = "\n\n".join(
            f"[{hit['metadata'].get('source', '?')}] {hit['text']}" for hit in hits
        )
    else:
        context = "\n\n".join(hit["text"] for hit in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def retrieve_context(question, collection, k=3, include_superseded=False):
    """The chunks sent to the LLM: wide hybrid retrieval, narrowed by the reranker."""
    candidates = hybrid_retrieve(collection, question, k=CANDIDATES, include_superseded=include_superseded)
    return rerank(question, candidates, top_n=k)


def answer(question, collection, llm, k=3, show_sources=True, include_superseded=False):
    """Retrieve context, ask llm (a prompt -> str callable), and report sources."""
    hits = retrieve_context(question, collection, k=k, include_superseded=include_superseded)
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
