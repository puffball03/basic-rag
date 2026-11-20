"""Deliverable 9: the question that surfaces the planted data-quality issue, the flawed answer,
the two-question diagnosis (retrieval or source data?), and the answer after the lineage fix.

Calls the Foundry LLM twice, so it needs FOUNDRY_* in .env. Full write-up: DIAGNOSIS.md.
"""
from chunker import load_docs, parse_header, split_chunks, superseded_sources
from rag import answer, build_index, make_foundry_llm, retrieve_context

QUESTION = "How long can a temporary increase in my remote days be approved for?"

docs = load_docs("./policies/")
collection = build_index(split_chunks(docs), "diagnosis_demo")
llm = make_foundry_llm()


def show_context(include_superseded):
    for hit in retrieve_context(QUESTION, collection, include_superseded=include_superseded):
        m = hit["metadata"]
        print(f"  {hit['id']:<28} {m['status']:<10} score={hit['score']:6.2f}  § {m['section']}")


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


print(f"Question: {QUESTION!r}")

banner("BEFORE THE FIX: every document is searchable (include_superseded=True)")
print("Retrieved context:")
show_context(include_superseded=True)
print("\nAnswer:")
print(" ", answer(QUESTION, collection, llm=llm, include_superseded=True)["answer"])

banner("DIAGNOSIS")
print("Q1. Was it retrieval?")
print("  No. The top chunk (v2 § 8. Requesting Changes) is exactly about temporary increases")
print("  in remote days. Retrieval returned the text that best matches the question.\n")
print("Q2. Was it the source data?")
for source in ("remote_work_policy_v2", "remote_work_policy_v3"):
    h = parse_header(docs[source])
    supersedes = h.get("supersedes", "-")
    print(f"  {source}: version {h['version']}, effective {h['effective date']}, supersedes: {supersedes}")
print("  Yes. The outdated v2 is still in the corpus, and nothing in its chunks says it was")
print("  replaced; only v3's header (not retrieved here) says so. v3 has no 'temporary increase'")
print("  process, because hybrid schedules were discontinued on September 1, 2026.")
print(f"\n  Lineage check, superseded_sources(): {sorted(superseded_sources(docs))}")

banner("AFTER THE FIX: superseded documents excluded (default)")
print("Retrieved context:")
show_context(include_superseded=False)
print("\nAnswer:")
print(" ", answer(QUESTION, collection, llm=llm)["answer"])
