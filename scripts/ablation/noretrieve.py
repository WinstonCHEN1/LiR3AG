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

    # indexed_contexts = "\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context_list)])

    prompt_template = '''
You are a helpful reasoning assistant.  Given a **Question** and think step‑by‑step.

* Provide steps of reasoning **based on your existing knowledge**.
* Write a concise, numbered chain‑of‑thought.  Begin every line with "Step X:" (starting from 1).
* **Output only the reasoning steps.**

### Example

Question: Were Scott Derrickson and Ed Wood of the same nationality?

Reasoning (desired output):
Step 1: We know Scott Derrickson is American.
Step 2: We know Ed Wood is American.
Step 3: Since both are American, they share the same nationality.

---
Question: {question}
Write your reasoning steps now. Remember: only the steps, nothing else.
'''

    prompt = prompt_template.format(question=question)

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
    filename = f"noretrieve_{dataset}_{method}_{safe_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")

    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "cuda:3"})
    # vectorstore = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)

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

        # docs = vectorstore.similarity_search(question, k=args.top_k)
        # contexts = [doc.page_content for doc in docs]
        contexts = []
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