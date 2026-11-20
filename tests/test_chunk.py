# test_chunk.py
from pathlib import Path

import pytest

from chunker import load_docs, split_chunks

POLICY_DIR = Path(__file__).parent.parent / "policies"
EXPECTED_POLICIES = {
    "pto_policy",
    "remote_work_policy_v2",
    "remote_work_policy_v3",
    "workplace_expense_policy",
}


# ---------- fixtures ----------

@pytest.fixture
def md_dir(tmp_path):
    (tmp_path / "alpha.md").write_text("# Alpha\n\n## Section\nAlpha body text.", encoding="utf-8")
    (tmp_path / "beta.md").write_text("# Beta\n\nBeta body text.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("should be ignored", encoding="utf-8")
    return tmp_path


@pytest.fixture
def policy_docs():
    return load_docs(POLICY_DIR)


@pytest.fixture
def policy_chunks(policy_docs):
    return split_chunks(policy_docs)


# ---------- ingestion: load_docs ----------

def test_load_docs_returns_dict(md_dir):
    assert isinstance(load_docs(md_dir), dict)


def test_load_docs_keys_are_file_stems(md_dir):
    assert set(load_docs(md_dir)) == {"alpha", "beta"}


def test_load_docs_ignores_non_markdown(md_dir):
    assert "notes" not in load_docs(md_dir)


def test_load_docs_values_are_full_file_text(md_dir):
    docs = load_docs(md_dir)
    assert docs["beta"] == "# Beta\n\nBeta body text."


def test_load_docs_accepts_str_path(md_dir):
    assert load_docs(str(md_dir)) == load_docs(md_dir)


def test_load_docs_empty_dir_returns_empty_dict(tmp_path):
    assert load_docs(tmp_path) == {}


def test_load_docs_preserves_utf8(tmp_path):
    (tmp_path / "intl.md").write_text("# Café — résumé ✓", encoding="utf-8")
    assert load_docs(tmp_path)["intl"] == "# Café — résumé ✓"


def test_load_docs_is_not_recursive(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "deep.md").write_text("# Deep", encoding="utf-8")
    assert load_docs(tmp_path) == {}


def test_load_docs_real_policy_corpus(policy_docs):
    assert set(policy_docs) == EXPECTED_POLICIES
    assert all(text.strip() for text in policy_docs.values())


# ---------- chunking: split_chunks ----------

def test_split_chunks_returns_nonempty_list(policy_chunks):
    assert isinstance(policy_chunks, list)
    assert len(policy_chunks) > 0


def test_chunks_have_rag_interface(policy_chunks):
    # rag.py relies on .page_content and .metadata
    for c in policy_chunks:
        assert isinstance(c.page_content, str)
        assert isinstance(c.metadata, dict)


def test_no_empty_chunks(policy_chunks):
    assert all(c.page_content.strip() for c in policy_chunks)


def test_more_chunks_than_documents(policy_docs, policy_chunks):
    # each ~4.5KB policy should split into several chunks
    assert len(policy_chunks) > len(policy_docs)


def test_every_chunk_has_source_metadata(policy_chunks):
    for c in policy_chunks:
        assert c.metadata.get("source") in EXPECTED_POLICIES


def test_every_document_is_represented(policy_chunks):
    assert {c.metadata["source"] for c in policy_chunks} == EXPECTED_POLICIES


def test_chunk_content_comes_from_its_source(policy_docs, policy_chunks):
    # ignore whitespace differences the splitter may introduce
    norm = lambda s: " ".join(s.split())
    for c in policy_chunks:
        assert norm(c.page_content) in norm(policy_docs[c.metadata["source"]])


def test_chunks_respect_size_limit(policy_chunks):
    MAX_CHARS = 1000  # set to your splitter's chunk_size
    assert all(len(c.page_content) <= MAX_CHARS for c in policy_chunks)


def test_v2_and_v3_chunks_stay_distinguishable(policy_chunks):
    # both remote-work docs share the same title; metadata must keep them apart
    sources = {c.metadata["source"] for c in policy_chunks}
    assert {"remote_work_policy_v2", "remote_work_policy_v3"} <= sources


def test_key_content_survives_chunking(policy_chunks):
    all_text = " ".join(c.page_content for c in policy_chunks)
    for phrase in ["Sick Leave", "Corporate Cards", "Return to Office", "Hybrid Work Model"]:
        assert phrase in all_text


def test_split_chunks_empty_input():
    assert split_chunks({}) == []


def test_short_doc_yields_single_chunk():
    chunks = split_chunks({"tiny": "# Tiny\n\nJust one line."})
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "tiny"


def test_split_chunks_is_deterministic(policy_docs):
    a = [(c.page_content, c.metadata) for c in split_chunks(policy_docs)]
    b = [(c.page_content, c.metadata) for c in split_chunks(policy_docs)]
    assert a == b