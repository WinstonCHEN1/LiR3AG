import os
import json
import time
import argparse
import requests
import datetime
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils import run_tokenization_experiment

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def call_ollama(question, context, model, think, api_url="http://localhost:11434/api/generate"):
    prompt = (
        "You are a helpful assistant that uses provided context to answer queries. "
        "If there is relevant information in the context provided, please follow it. "
        "If there is really no relevant content in the relevant context, please generate it yourself."
        "But please **don't explain the answer** anyway, just answer.\n\n"

        "Example 1:\n"
        "    Question: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?\n"
        "    Answer(No explain): Sam Bankman-Fried\n"

        "Example 2:\n"
        "    Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
        "    Answer(No explain): Yes\n\n"

        f"Question: {question}\n\n"
        f"Context: {context}\n\n"

        "If it is not a judgment type, just output the answer as in Example 1.\n"
        "If your answer is a yes or no judgment, just output Yes or No like Example 2.\n"
        "All in all, **do not explain the answer** anyway!"
        "Please output the Answer."
    )
    if not think:
        prompt += "/no_think"
    try:
        response = requests.post(api_url, json={
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "stream": False
        }, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama ERROR] {e}")
        return ""


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="OLLAMA API RAG Process")
    parser.add_argument("--input", required=True, help="input jsonl")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--index_dir", default="../../index", help="LangChain saved index directory")
    parser.add_argument("--model", default="llama3.1:8b", help="model name")
    parser.add_argument("--top_k", type=int, default=5, help="top_k")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--think", dest="think", action="store_true", help="think mode")
    group.add_argument("--nothink", dest="think", action="store_false", help="non-think mode")
    parser.set_defaults(think=True)
    args = parser.parse_args()
    if args.think:
        think = True
        print("think enabled")
    else:
        think = False
        print("think disabled")
    
    output_dir = args.output
    dataset = os.path.splitext(os.path.basename(args.input))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method = os.path.basename(current_dir)
    model = args.model
    safe_name = model.replace(":", "-")
    if think:
        model_name = safe_name + "_think"
    else:
        model_name = safe_name + "_nothink"
    filename = f"{dataset}_{method}_{model_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)

    questions = read_jsonl(args.input)
    results = []
    input_texts = []
    output_texts = []

    for item in tqdm(questions, desc="Processing"):
        qid = item.get("id")
        question = item.get("question", "").strip()
        if not qid or not question:
            continue

        docs = vectorstore.similarity_search(question, k=args.top_k)
        context = "\n---\n".join([doc.page_content for doc in docs])

        answer = call_ollama(question, context, args.model, think=think)
        results.append({"id": qid, "generate": answer})

        full_input = f"Question: {question}\nContext: {context}"
        input_texts.append(full_input)
        output_texts.append(answer)

    write_jsonl(output_path, results)
    print(f"[FINISH] Output saved：{output_path}")
    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[LOG] Token log saved：{log_path}")


if __name__ == "__main__":
    main()

# python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --index_dir "../../index/HotpotQA" --model qwen3:14b --top_k 5 --think