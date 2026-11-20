# basic-rag

An internal Q&A assistant over Arasaka's company policy documents (`policies/`), built as a retrieval-augmented generation (RAG) pipeline: `chunker.py` splits the documents, `rag.py` embeds and stores them in chromadb, retrieves the most relevant chunks for a question, and asks an LLM to answer from them.

## Chunking strategy

Implemented in [`chunker.py`](chunker.py) in two passes:

1. **Split on markdown headers** (`#` title, `##` section) with `MarkdownHeaderTextSplitter`, keeping the header text in the chunk.
2. **Split anything still too long** with `RecursiveCharacterTextSplitter`: `CHUNK_SIZE = 1000` characters, `CHUNK_OVERLAP = 150`.

### Why split on headers first

The policies are structured as numbered sections ("5. Sick Leave", "4. Approval Thresholds"), and each section answers one kind of question. Splitting on headers keeps a section's rules together instead of cutting mid-rule at an arbitrary character count. It also gives every chunk its `title` and `section` metadata for free, which source attribution uses.

### Why 1000 characters

Measured on the current corpus (4 documents, ~700–750 words each, 45 sections):

| | Value |
|---|---|
| Median section length | 364 chars |
| Longest section | 1,347 chars (remote work v3, "3. Sick-Day Remote Work Exception") |
| Sections over 1,000 chars | 1 of 45 |
| Resulting chunks | 46, median 370 chars, max 954 |

- **Most chunks are exactly one section.** 44 of 45 sections fit under 1,000 characters, so the size limit only cuts the one unusually long section. A smaller limit (say 300–500) would split many ordinary sections in half and separate rules from the heading that gives them meaning.
- **It stays inside the embedding model's limit.** `all-MiniLM-L6-v2` reads at most 256 tokens and silently ignores anything past that. The longest chunk is 228 tokens, so every chunk is embedded in full. Much above ~1,100 characters, the ends of chunks would start being dropped from their embeddings.
- **Small chunks keep retrieval precise.** Each chunk is about one topic, so the top-k results sent to the LLM are mostly relevant text rather than whole documents.

### Why 150 overlap

The overlap only matters when a section is split, which currently happens once. When it does, 150 characters (about 15% of a chunk, or one to two bullet points) repeats the end of one piece at the start of the next. A rule that straddles the cut, such as a bullet list of conditions, then appears whole in at least one chunk. Much larger overlap would mostly duplicate text, and with this corpus it would buy nothing.

### Revisit if

- New documents have much longer sections: more splits mean the overlap starts to matter.
- The embedding model changes: re-check its token limit against `CHUNK_SIZE`.
