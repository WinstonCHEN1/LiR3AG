import os
import json
import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def build_faiss_index(corpus_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load documents
    raw_data = read_jsonl(corpus_path)
    documents = []
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "。", "!", "！", "?", "？", " ", ""]
    )

    for item in tqdm(raw_data, desc="Splitting text and building documents"):
        content = item["text"]
        metadata = {"id": item["id"]}

        # Split text chunks
        chunks = text_splitter.split_text(content)
        
        # Wrap each chunk into a Document, preserving metadata
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"id": item["id"], "chunk_id": i}
            ))

    # Step 2: Embedding model
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda:0"}  # or "cpu"
    )
    # Step 3: Build FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    # Step 4: Save vectorstore
    vectorstore.save_local(output_dir)
    print(f"FAISS index saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index with text")
    parser.add_argument("--corpus", required=True, help="Input JSONL corpus file containing id and text fields")
    parser.add_argument("--output_dir", required=True, help="Index root directory")
    args = parser.parse_args()

    # Get JSONL filename (without extension) as subdirectory name
    corpus_filename = os.path.splitext(os.path.basename(args.corpus))[0]
    sub_output_dir = os.path.join(args.output_dir, corpus_filename)
    build_faiss_index(args.corpus, sub_output_dir)

if __name__ == "__main__":
    main()
# python build_faiss_index.py --corpus "../corpus/HotpotQA.jsonl" --output_dir "../index"