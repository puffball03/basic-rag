# test_eval.py: Part 5 evaluation harness (retrieval recall + answer accuracy)
import os
import re
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

import rag
from chunker import load_docs, split_chunks
from eval_set import EVAL_SET

POLICY_DIR = Path(__file__).parent.parent / "policies"
K = 3

load_dotenv()  # local runs pick up .env; CI has no key, so LLM evals skip
needs_llm = pytest.mark.skipif(not os.getenv("FOUNDRY_API_KEY"), reason="FOUNDRY_API_KEY not set")


@pytest.fixture(scope="module")
def collection():
    chunks = split_chunks(load_docs(POLICY_DIR))
    return rag.build_index(chunks, f"eval_{uuid.uuid4().hex[:8]}")


def case_id(case):
    return case["question"]


@pytest.mark.parametrize("case", EVAL_SET, ids=case_id)
def test_retrieval_recall(collection, case):
    """Did the chunk holding the answer reach the LLM's context?"""
    hits = rag.retrieve_context(case["question"], collection, k=K)
    retrieved = [(h["metadata"]["source"], h["metadata"]["section"]) for h in hits]
    expected = (case["expected_source"], case["expected_section"])
    assert expected in retrieved, f"expected {expected}, retrieved {retrieved}"


@pytest.fixture(scope="module")
def llm():
    return rag.make_foundry_llm()


def missing_facts(answer, fact_groups):
    """Fact groups with none of their alternatives in the answer (case-insensitive).

    Bold markers and the policies' "two (2)" numeral style are stripped first,
    so "**two (2) weeks**" matches "two weeks".
    """
    text = re.sub(r"\*\*|\s*\(\d+\)", "", answer).lower()
    return [group for group in fact_groups if not any(alt.lower() in text for alt in group)]


@pytest.mark.llm
@needs_llm
@pytest.mark.parametrize("case", EVAL_SET, ids=case_id)
def test_answer_accuracy(collection, llm, case):
    """Does the generated answer contain the expected key facts?"""
    result = rag.answer(case["question"], collection, llm=llm, k=K)
    missing = missing_facts(result["answer"], case["expected_facts"])
    assert not missing, f"missing {missing} in answer: {result['answer']!r}"
