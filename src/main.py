import os
import time
import argparse
import datetime
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from reasoning import generate_reasoning_steps
from api_calls import call_deepseek, call_ollama
from utils import run_tokenization_experiment, read_jsonl, write_jsonl

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="RAG reasoning system based on FAISS and LLM")
    parser.add_argument("--input", required=True, help="Input JSONL file containing id and question fields")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--index_dir", default="../../index", help="Directory for FAISS index saved by LangChain")
    parser.add_argument("--model", default="qwen3:8b", help="Model name")
    parser.add_argument("--top_k", type=int, default=5, help="Number of contexts to retrieve per question")
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

    # Initialize embeddings and vectorstore
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
            "Please output the answer.")

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
        input_texts.append(prompt)
        output_texts.append(answer)

    write_jsonl(output_path, results)
    print(f"[Completed] Results saved to: {output_path}")

    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[Log] Token statistics saved to: {log_path}")

if __name__ == "__main__":
    main()