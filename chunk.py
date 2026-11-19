from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path



def load_docs(path):
    files = {}
    md_dir = Path(path)
    
    for file_path in md_dir.glob("*.md"):
        files[file_path.stem] = file_path.read_text(encoding="utf-8")
    return files

def split_chunks(files):
    #md_splitter = MarkdownHeaderTextSplitter()
    for file in files:

