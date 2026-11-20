"""Deliverable 5: a query where hybrid retrieval beats vector-only search."""
from chunker import load_docs, split_chunks
from rag import build_index, hybrid_retrieve, retrieve

QUESTION = "What is HR-201?"
EXPECTED = "pto_policy::0"  # the PTO policy header chunk, the only one containing "HR-201"
K = 3

collection = build_index(split_chunks(load_docs("./policies/")), "hybrid_demo")


def show(label, hits):
    print(f"\n{label} (top {K}):")
    for rank, hit in enumerate(hits, start=1):
        marker = "  <-- expected" if hit["id"] == EXPECTED else ""
        print(f"  {rank}. {hit['id']}{marker}")
    found = any(hit["id"] == EXPECTED for hit in hits)
    print(f"  found {EXPECTED}: {'YES' if found else 'NO'}")


# Both searches see the full corpus, superseded remote_work_policy_v2 included, so the
# comparison matches the original Deliverable 5 run (before the Part 6 lineage fix).
print(f"Query: {QUESTION!r}")
show("Vector-only", retrieve(collection, QUESTION, k=K, include_superseded=True))
show("Hybrid (vector + keyword, RRF)", hybrid_retrieve(collection, QUESTION, k=K, include_superseded=True))
