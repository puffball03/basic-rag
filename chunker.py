import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
ID_SEPARATOR = "::"

HEADERS_TO_SPLIT_ON = [("#", "title"), ("##", "section")]

HEADER_FIELD = re.compile(r"^\*\*(.+?):\*\*\s*(.+)$", re.MULTILINE)


def load_docs(path):
    """Read every top-level *.md file in path and return."""
    files = {}
    for file_path in Path(path).glob("*.md"):
        files[file_path.stem] = file_path.read_text(encoding="utf-8")
    return files


def parse_header(text):
    """Bold header fields, e.g. {'policy number': 'HR-104', 'version': '3.0', 'supersedes': ...}."""
    return {key.lower(): value.strip() for key, value in HEADER_FIELD.findall(text)}


def superseded_sources(files):
    """Sources whose policy number + version is named in another document's 'Supersedes' field."""
    headers = {source: parse_header(text) for source, text in files.items()}
    replaced = [h["supersedes"] for h in headers.values() if "supersedes" in h]
    return {
        source
        for source, h in headers.items()
        if any(r.startswith(f"{h.get('policy number')} Version {h.get('version')}") for r in replaced)
    }


def split_chunks(files):
    """Split on section seperators, then by character count with overlap."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    superseded = superseded_sources(files)
    chunks = []
    for source in sorted(files):
        sections = header_splitter.split_text(files[source])
        pieces = size_splitter.split_documents(sections)
        pieces = [p for p in pieces if p.page_content.strip()]

        for index, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=piece.page_content,
                    metadata={
                        "id": f"{source}{ID_SEPARATOR}{index}",
                        "source": source,
                        "title": piece.metadata.get("title", ""),
                        "section": piece.metadata.get("section", ""),
                        "chunk_index": index,
                        "status": "superseded" if source in superseded else "current",
                    },
                )
            )
    return chunks
