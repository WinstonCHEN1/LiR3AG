import os
import json
import time
import requests
import argparse
import datetime
from tqdm import tqdm

from utils import run_tokenization_experiment

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def call_ollama(prompt, model, api_url):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Generate failed: {e}")
        return ""

def process(input_file, output_dir, model, think):
    start_time = time.time()
    data = read_jsonl(input_file)
    results = []
    input_texts = []
    output_texts = []

    api_url = "http://localhost:11434/api/generate"

    for entry in tqdm(data, desc="Generating"):
        qid = entry.get("id")
        question = entry.get("question", "").strip()
        if not qid or not question:
            continue

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

            "If it is not a judgment type, just output the answer as in Example 1.\n"
            "If your answer is a yes or no judgment, just output Yes or No like Example 2.\n"
            "All in all, **do not explain the answer** anyway!"
            "Please output the Answer."
        )

        if not think:
            prompt += "/no_think"

        answer = call_ollama(prompt, model, api_url)

        results.append({
            "id": qid,
            "generate": answer
        })
        input_texts.append(prompt)
        output_texts.append(answer)

    dataset = os.path.splitext(os.path.basename(input_file))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method = os.path.basename(current_dir)
    safe_name = model.replace(":", "-")
    if think:
        model_name = safe_name + "_think"
    else:
        model_name = safe_name + "_nothink"
    filename = f"{dataset}_{method}_{model_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")
    
    write_jsonl(output_path, results)
    print(f"[FINISH] Output saved：{output_path}")
    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[LOG] Token log saved：{log_path}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Ollama qwen3 API requests")
    parser.add_argument("--input", "-i", required=True, help="input jsonl")
    parser.add_argument("--output", "-o", required=True, help="output directory")
    parser.add_argument("--model", "-m", default="qwen3:8b", help="ollama model name")
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
    
    process(args.input, args.output, args.model, think=think)

if __name__ == "__main__":
    main()

# python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --model qwen3:8b --think
