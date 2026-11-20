# test_rag.py
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag
from chunker import load_docs, split_chunks

POLICY_DIR = Path(__file__).parent.parent / "policies"


# ---------- helpers / fixtures ----------

def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_chunk(chunk_id, text, source):
    return Document(
        page_content=text,
        metadata={"id": chunk_id, "source": source, "title": "", "section": "", "chunk_index": 0},
    )


@pytest.fixture
def tiny_chunks():
    return [
        make_chunk("pets::0", "Employees may bring dogs to the office on Fridays.", "pets"),
        make_chunk("parking::0", "Parking permits cost forty dollars per month.", "parking"),
    ]


@pytest.fixture
def tiny_collection(tiny_chunks):
    return rag.build_index(tiny_chunks, unique_name("tiny"))


@pytest.fixture(scope="module")
def policy_chunks():
    return split_chunks(load_docs(POLICY_DIR))


@pytest.fixture(scope="module")
def policy_collection(policy_chunks):
    return rag.build_index(policy_chunks, unique_name("policies"))


class FakeLLM:
    """Records the prompt it was given and returns a canned reply."""

    def __init__(self, reply="fake answer"):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


# ---------- indexing ----------

def test_build_index_stores_every_chunk(tiny_chunks, tiny_collection):
    assert tiny_collection.count() == len(tiny_chunks)


def test_build_index_full_corpus(policy_chunks, policy_collection):
    assert policy_collection.count() == len(policy_chunks)


# ---------- retrieval ----------

def test_retrieve_returns_k_hits(policy_collection):
    assert len(rag.retrieve(policy_collection, "How much PTO do I get?", k=3)) == 3


def test_retrieve_hit_shape(tiny_collection):
    for hit in rag.retrieve(tiny_collection, "dogs", k=2):
        assert isinstance(hit["id"], str)
        assert isinstance(hit["text"], str) and hit["text"]
        assert "source" in hit["metadata"]


def test_retrieve_ranks_relevant_text_first(tiny_collection):
    hits = rag.retrieve(tiny_collection, "Can I bring my pet to work?", k=2)
    assert hits[0]["id"] == "pets::0"


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("How many sick days do employees get?", "pto_policy"),
        ("What is the approval threshold for expenses?", "workplace_expense_policy"),
    ],
)
def test_retrieve_finds_right_document(policy_collection, question, expected_source):
    hits = rag.retrieve(policy_collection, question, k=3)
    assert expected_source in {h["metadata"]["source"] for h in hits}


# ---------- prompt ----------

def test_build_prompt_contains_question_and_context(tiny_chunks):
    hits = [{"id": c.metadata["id"], "text": c.page_content, "metadata": c.metadata} for c in tiny_chunks]
    prompt = rag.build_prompt("Can I bring my dog?", hits)
    assert "Can I bring my dog?" in prompt
    for c in tiny_chunks:
        assert c.page_content in prompt


def test_build_prompt_can_hide_source_labels(tiny_chunks):
    hits = [{"id": c.metadata["id"], "text": c.page_content, "metadata": c.metadata} for c in tiny_chunks]
    prompt = rag.build_prompt("Can I bring my dog?", hits, show_sources=False)
    assert "[pets]" not in prompt and "[parking]" not in prompt
    for c in tiny_chunks:
        assert c.page_content in prompt


# ---------- answer (end to end, fake LLM) ----------

def test_answer_returns_llm_output(tiny_collection):
    result = rag.answer("Can I bring my dog?", tiny_collection, llm=FakeLLM("Yes, on Fridays."))
    assert result["answer"] == "Yes, on Fridays."


def test_answer_sends_retrieved_context_to_llm(tiny_collection):
    llm = FakeLLM()
    rag.answer("Can I bring my dog?", tiny_collection, llm=llm, k=1)
    assert len(llm.prompts) == 1
    assert "dogs to the office on Fridays" in llm.prompts[0]


def test_answer_reports_sources(tiny_collection):
    result = rag.answer("Can I bring my dog?", tiny_collection, llm=FakeLLM(), k=1)
    assert result["sources"] == ["pets"]


def test_answer_with_hidden_sources_still_reports_them(tiny_collection):
    # Part 6: the LLM is blind to sources, but we still know where its context came from
    llm = FakeLLM()
    result = rag.answer("Can I bring my dog?", tiny_collection, llm=llm, k=1, show_sources=False)
    assert "[pets]" not in llm.prompts[0]
    assert "dogs to the office on Fridays" in llm.prompts[0]
    assert result["sources"] == ["pets"]


# ---------- keyword search (Part 3) ----------

def test_keyword_search_ranks_exact_code_first(policy_collection):
    hits = rag.keyword_search(policy_collection, "What is HR-201?", k=3)
    assert hits[0]["id"] == "pto_policy::0"


def test_keyword_search_no_matches_returns_empty(tiny_collection):
    assert rag.keyword_search(tiny_collection, "quantum xylophone", k=3) == []


# ---------- hybrid retrieval (Part 3) ----------

def test_hybrid_retrieve_returns_k_unique_hits(policy_collection):
    # "Sick Leave" is found by both searches, so the merge must de-duplicate
    hits = rag.hybrid_retrieve(policy_collection, "Sick Leave", k=3)
    ids = [h["id"] for h in hits]
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_hybrid_finds_policy_code_that_vector_search_misses(policy_collection):
    # Deliverable 5: embeddings don't know what "HR-201" means; keyword match does.
    # Compared on the full corpus (superseded v2 included), as in the Deliverable 5 screenshots:
    # v2's look-alike header chunks are what push the PTO header out of vector search's top 3.
    question = "What is HR-201?"
    vector_ids = [h["id"] for h in rag.retrieve(policy_collection, question, k=3, include_superseded=True)]
    hybrid_ids = [
        h["id"] for h in rag.hybrid_retrieve(policy_collection, question, k=3, include_superseded=True)
    ]
    assert "pto_policy::0" not in vector_ids
    assert "pto_policy::0" in hybrid_ids


# ---------- reranking (Part 4) ----------

def test_rerank_returns_top_n_sorted_by_score(policy_collection):
    candidates = rag.hybrid_retrieve(policy_collection, "How many sick days do employees get?", k=10)
    reranked = rag.rerank("How many sick days do employees get?", candidates, top_n=3)
    scores = [h["score"] for h in reranked]
    assert len(reranked) == 3
    assert scores == sorted(scores, reverse=True)


def test_rerank_fixes_wrong_input_order(tiny_chunks):
    hits = [{"id": c.metadata["id"], "text": c.page_content, "metadata": c.metadata} for c in tiny_chunks]
    hits.reverse()  # parking first: the wrong order for this question
    reranked = rag.rerank("Can I bring my dog to work?", hits, top_n=2)
    assert reranked[0]["id"] == "pets::0"


def test_rerank_with_fewer_hits_than_top_n_returns_all(tiny_chunks):
    hits = [{"id": c.metadata["id"], "text": c.page_content, "metadata": c.metadata} for c in tiny_chunks]
    assert len(rag.rerank("parking", hits, top_n=5)) == 2


# ---------- final context sent to the LLM ----------

def test_retrieve_context_returns_k_reranked_hits(policy_collection):
    hits = rag.retrieve_context("How many sick days do employees get?", policy_collection, k=3)
    assert len(hits) == 3
    assert all("score" in h for h in hits)


# ---------- superseded documents (Part 6 fix) ----------

V2_ONLY_QUESTION = "How long can a temporary increase in my remote days be approved for?"


def test_retrieve_context_skips_superseded_documents(policy_collection):
    hits = rag.retrieve_context(V2_ONLY_QUESTION, policy_collection, k=3)
    assert all(h["metadata"]["status"] == "current" for h in hits)


def test_retrieve_context_can_include_superseded_documents(policy_collection):
    hits = rag.retrieve_context(V2_ONLY_QUESTION, policy_collection, k=3, include_superseded=True)
    assert "remote_work_policy_v2::8" in [h["id"] for h in hits]
