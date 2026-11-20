# Part 6 — Diagnosing the planted data-quality issue

Reproduce with `python demo_diagnosis.py` (needs the `FOUNDRY_*` settings in `.env`).

## The planted issue

The corpus holds two versions of the same policy, **HR-104 Remote Work Policy**:

| File | Version | Effective | Says |
|---|---|---|---|
| `remote_work_policy_v2.md` | 2.0 | March 1, 2025 | Hybrid model: up to 3 remote days a week, anchor days, $500 home-office stipend, $50/month internet, temporary increases in remote days for up to 4 weeks |
| `remote_work_policy_v3.md` | 3.0 | September 1, 2026 | Full return to office, 5 days a week. The only remote work allowed is a sick-day exception. Stipend and internet reimbursement discontinued |

v2 is an **outdated duplicate** that was never removed. The only record that it was replaced is one line in v3's header: `**Supersedes:** HR-104 Version 2.0 (Hybrid Work Model)`.

## Finding a question that exposes it

**The obvious conflict questions did not fail.** I asked "How many days a week do I need to work from the office?", "Can I still get the home office stipend?" and "Do I get reimbursed for my home internet?", with source labels shown and hidden. The model answered correctly every time: five days, and the stipend and internet reimbursement are discontinued.

The reason is that v3's body names what it replaces ("The hybrid model introduced in Version 2.0 … is discontinued"; "The one-time home office stipend and the monthly internet reimbursement offered under Version 2.0 are discontinued"). For these questions, that v3 text is retrieved next to the conflicting v2 text, so the context itself resolves the conflict. Answers sometimes hedged ("Before September 1, 2026, the hybrid policy applies…"), but none were wrong.

**The questions that fail are the ones only v2 answers.** If v3 says nothing on a topic, nothing in the retrieved context contradicts the outdated rule:

> **Q:** How long can a temporary increase in my remote days be approved for?
>
> **A (flawed):** "According to the Remote Work Policy (Section 8, Requesting Changes), a temporary increase in remote days, such as for a personal or family situation, may be approved for up to **four (4) weeks** at a time."

That is wrong under the current policy. Since September 1, 2026, everyone works in the office five days a week. There are no remote days to increase. The only remote work allowed is the sick-day exception (at most 3 consecutive days, 10 per year).

## The two-question diagnosis

### Q1. Was it retrieval?

**No.** Retrieved context (hybrid retrieval → cross-encoder rerank, top 3):

| Chunk | Status | Rerank score | Section |
|---|---|---|---|
| `remote_work_policy_v2::8` | superseded | 8.58 | 8. Requesting Changes |
| `remote_work_policy_v2::2` | superseded | −2.38 | 2. Hybrid Work Model |
| `remote_work_policy_v2::3` | superseded | −3.04 | 3. Eligibility |

The top chunk is about exactly what was asked: requesting a temporary increase in remote days. It scores far above everything else. Retrieval did what it is designed to do and returned the text in the corpus that best matches the question.

The generation step also behaved correctly. The model answered faithfully from the context it was given and even cited the section. The answer is wrong because the context is outdated, not because the model misread it.

### Q2. Was it the source data?

**Yes.** Three facts about the data together explain the failure:

1. **The outdated document is still in the corpus**, indexed and searchable like any current document.
2. **Nothing in v2's chunks marks it as outdated.** Its header says "Version 2.0, effective March 1, 2025", but that header is a separate chunk (`::0`) and isn't retrieved here. The retrieved chunks are just policy text.
3. **The only lineage record is in v3's header** (`Supersedes: HR-104 Version 2.0`), and that chunk isn't retrieved for this question either. v3 has no section on temporary increases, so no v3 text competes with v2 § 8.

### Confirming it

The same code, with only the data changed:

- **v2 removed from the corpus:** the answer becomes "The provided context doesn't mention any process for temporarily increasing remote days. Under the current policy, effective September 1, 2026, employees must work from their assigned Arasaka office five days a week. The only exception allowing remote work is the sick-day remote work exception…"
- **Source labels shown (`show_sources=True`) with v2 still present:** the model still says "four weeks". A label like `remote_work_policy_v2` doesn't tell the model the document is outdated when no v3 chunk is in the context to compare it with. Labelling sources is not enough on its own.

Changing the data fixes the answer and changing the prompt doesn't. That confirms the root cause is the source data, not retrieval or generation.

## Root cause

**An outdated duplicate policy was left in the corpus with no lineage on its own content.** Retrieval ranks text by how relevant it is to the question, not by whether it is current. So whenever v2 is the only document addressing a topic, its outdated rule is served as fact.

## The fix

Make lineage part of the data, and filter on it at retrieval time. This was built test-first; see `tests/test_chunk.py` and `tests/test_rag.py`.

1. **`chunker.py`:**
   - `parse_header()` reads each document's bold header fields.
   - `superseded_sources()` marks a document superseded when another document's `Supersedes` field names its policy number and version.
   - Every chunk gets `"status": "superseded"` or `"current"` in its metadata.
2. **`rag.py`:**
   - Vector search and keyword search filter on `{"status": "current"}` by default, so superseded chunks never reach the LLM.
   - `include_superseded=True` brings back the old behavior: for the "before" side of this demo, or to look up what an old policy said on purpose.
3. v2 stays **in the index**, so the history is still there for auditing.

After the fix, the same question retrieves v3 § 2 Return to Office, PTO § 5 Sick Leave and v3 § 3 Sick-Day Remote Work Exception. The answer becomes: "The provided context doesn't say how long a temporary increase in remote days can be approved for… effective September 1, 2026, all employees must work from their assigned office five days per week… The only remote work it allows is the sick-day exception…"

The question is in the evaluation set (`tests/eval_set.py`), so retrieval recall and answer accuracy both check for it from now on.

## Limitations

- **The fix only works when documents declare what they replace.** A stale copy with no `Supersedes` line, or one written in a different format, would get through. A higher version number alone is deliberately not treated as superseding (`test_documents_are_current_when_nothing_supersedes_them`): replacement has to be declared, not guessed.
- **Side effect on the Part 3 demo.** With v2's look-alike header chunks filtered out, vector-only search ranks the HR-201 header 3rd instead of missing it. The hybrid demo (`demo_hybrid.py`) therefore compares both searches on the full corpus (`include_superseded=True`), matching the original Deliverable 5 run.

## Lesson

Retrieval can't tell current text from outdated text unless the data says which is which. When documents are versioned, lineage (which version replaces which) has to be stored with the content and enforced at retrieval time. Otherwise the pipeline will confidently serve whichever version happens to match the question best.
