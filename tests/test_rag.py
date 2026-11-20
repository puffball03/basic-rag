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
