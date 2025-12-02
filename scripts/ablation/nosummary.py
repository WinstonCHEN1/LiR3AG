import os
import ast
import json
import time
import argparse
import requests
import datetime
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from utils import run_tokenization_experiment

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def generate_reasoning_steps(question, context_list, model, client, api_url="http://localhost:11434/api/generate"):
    """Return a string containing numbered reasoning steps (CoT)."""

    indexed_contexts = "\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context_list)])

    prompt_template = '''

You are a helpful reasoning assistant. Given a **Question** and several **Contexts**, your task is to **rank all the contexts in order of relevance to the question**, from most relevant to least relevant.
* Do not discard any contexts.
* Do not modify the content of the contexts — preserve the original text exactly.
* Output the list of contexts in ranked order by relevance.
### Example
Contexts
1. Scott Derrickson is an American film director.
2. Bananas are rich in potassium.
3. Ed Wood (1924‑1978) was an American filmmaker.
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Output:
1. Scott Derrickson is an American film director.
2. Ed Wood (1924‑1978) was an American filmmaker.
3. Bananas are rich in potassium.
---
{indexed_contexts}
Question: {question}
Please rank all contexts in order of relevance to the question, and **remember to keep the original text**.

'''

    prompt = prompt_template.format(indexed_contexts=indexed_contexts, question=question)

    if model == "deepseek-chat":
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DeepSeek Error - Reasoning Generation] {e}")
            return ""
    else:
        # try:
        #     response = requests.post(api_url, json={
        #         # "model": model,
        #         "model": "qwen3:32b",
        #         "prompt": prompt,
        #         "temperature": 0,
        #         "stream": False
        #     }, timeout=120)
        #     response.raise_for_status()
        #     return response.json().get("response", "").strip()
        # except Exception as e:
        #     print(f"[Ollama Error - Reasoning Generation] {e}")
        #     return ""
        try:
            os.environ["OPENAI_API_KEY"] = "sk-..."
            gptclient = OpenAI()
            response = gptclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI Error - Reasoning Generation] {e}")
            return ""

def call_deepseek(question, context, prompt, model, client):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek Error] Request Failed: {e}")
        return ""

def call_ollama(question, context, prompt, model, api_url="http://localhost:11434/api/generate"):
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
        print(f"[Ollama Error - Answer Generation] {e}")
        return ""

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="RAG Reasoning System Based on FAISS + LLM")
    parser.add_argument("--input", required=True, help="Input JSONL file containing id and question fields")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--index_dir", default="../../index", help="Directory for FAISS index saved by LangChain")
    parser.add_argument("--model", default="qwen3:8b", help="Model name")
    parser.add_argument("--top_k", type=int, default=5, help="Number of contexts retrieved per question")
    args = parser.parse_args()

    output_dir = args.output
    dataset = os.path.splitext(os.path.basename(args.input))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method = os.path.basename(current_dir)
    model = args.model
    safe_name = model.replace(":", "-")
    filename = f"nosummary_{dataset}_{method}_{safe_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "cuda:3"})
    vectorstore = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)

    questions = read_jsonl(args.input)
    results = []
    input_texts = []
    output_texts = []

    client = None
    if model == "deepseek-chat":
        os.environ["DEEPSEEK_API_KEY"] = "sk-..."
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for item in tqdm(questions, desc="Processing"):
        qid = item.get("id")
        question = item.get("question", "").strip()
        if not qid or not question:
            continue

        docs = vectorstore.similarity_search(question, k=args.top_k)
        contexts = [doc.page_content for doc in docs]

        reasoning_steps = generate_reasoning_steps(question, contexts, model, client)
        final_context = reasoning_steps

        prompt = (
            "You are a helpful assistant that uses provided context to answer queries. "
            "If there is relevant information in the context provided, please follow it. "
            "If there is really no relevant content in the relevant context, please generate it yourself."
            "But please **don't explain the answer** anyway, just answer.\n\n"

            "Example 1:\n"
            "Question: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?\n"
            "Answer(No explain): Sam Bankman-Fried\n"

            "Example 2:\n"
            "Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
            "Answer(No explain): Yes\n\n"

            f"Question: {question}\n\n"
            f"Context: {final_context}\n\n"

            "If it is not a judgment type, just output the answer as in Example 1.\n"
            "If your answer is a yes or no judgment, just output Yes or No like Example 2.\n"
            "All in all, **do not explain the answer** anyway!"
            "Please output the answer."
            "/no_think")

        if model == "deepseek-chat":
            answer = call_deepseek(question, final_context, prompt, model, client)
        else:
            answer = call_ollama(question, final_context, prompt, model)

        results.append({
            "id": qid,
            "generate": answer,
            "reasoning_steps": reasoning_steps
        })

        full_input = f"Question: {question}\nReasoning:\n{reasoning_steps}"
        input_texts.append(full_input)
        output_texts.append(answer)

    write_jsonl(output_path, results)
    print(f"[Completed] Output results saved to: {output_path}")

    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[Log] Token statistics saved to: {log_path}")

if __name__ == "__main__":
    main()