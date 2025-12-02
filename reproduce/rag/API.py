import os
import json
import time
import argparse
import datetime
from tqdm import tqdm
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils import run_tokenization_experiment

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
def call_deepseek(question, context, client, model):
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
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        if model == "deepseek-reasoner":
            reasoning = response.choices[0].message.reasoning_content.strip()
            content = response.choices[0].message.content.strip()
            final_output = f"<think>{reasoning}</think>\n{content}"
        else:
            final_output = response.choices[0].message.content.strip()
        return final_output
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return ""

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="DS API RAG Process")
    parser.add_argument("--input", required=True, help="input jsonl")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--index_dir", default="../../index", help="LangChain saved index directory")
    parser.add_argument("--model", default="deepseek-chat", help="model name")
    parser.add_argument("--top_k", type=int, default=5, help="top_k")
    args = parser.parse_args()

    output_dir = args.output
    dataset = os.path.splitext(os.path.basename(args.input))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method = os.path.basename(current_dir)
    model = args.model
    safe_name = model.replace(":", "-")
    filename = f"{dataset}_{method}_{safe_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")

    os.environ["DEEPSEEK_API_KEY"] = "sk-..."
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.load_local(args.index_dir, embedder, allow_dangerous_deserialization=True)

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
        answer = call_deepseek(question, context, client, args.model)

        results.append({"id": qid, "generate": answer})

        input_texts.append(f"Question: {question}\nContext: {context}")
        output_texts.append(answer)

    write_jsonl(output_path, results)
    print(f"[FINISH] Output saved：{output_path}")

    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[LOG] Token log saved：{log_path}")

if __name__ == "__main__":
    main()

# python API.py --input "../../query/HotpotQA.jsonl" --output "../data" --index_dir "../../index/HotpotQA" --model deepseek-reasoner --top_k 5
