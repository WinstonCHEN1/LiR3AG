import os
import json
import time
import argparse
import datetime
from tqdm import tqdm
from openai import OpenAI

from utils import run_tokenization_experiment

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def call_openai(client, model, messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": messages},
            ],
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

def process(input_file, output_dir, client, model):
    start_time = time.time()
    data = read_jsonl(input_file)
    results = []
    input_texts = []
    output_texts = []

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

        answer = call_openai(client, model, prompt)

        results.append({"id": qid, "generate": answer})
        input_texts.append(prompt)
        output_texts.append(answer)

    dataset = os.path.splitext(os.path.basename(input_file))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method = os.path.basename(current_dir)
    safe_name = model.replace(":", "-")
    filename = f"{dataset}_{method}_{safe_name}_{timestamp}.jsonl"
    output_path = os.path.join(output_dir, filename)
    log_path = output_path.replace(".jsonl", ".log")

    write_jsonl(output_path, results)
    print(f"[FINISH] Output saved：{output_path}")
    end_time = time.time()
    run_tokenization_experiment(input_texts, output_texts, model_name=model, log_file_name=log_path, total_duration=end_time - start_time)
    print(f"[LOG] Token log saved：{log_path}")


def main():
    parser = argparse.ArgumentParser(description="API requests")
    parser.add_argument("--input", "-i", required=True, help="input jsonl")
    parser.add_argument("--output", "-o", required=True, help="output directory")
    parser.add_argument("--model", "-m", default="deepseek-chat", help="model name")

    args = parser.parse_args()
    os.environ["DEEPSEEK_API_KEY"] = "sk-..."

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    process(args.input, args.output, client, args.model)


if __name__ == "__main__":
    main()

# python API.py --input "../../query/HotpotQA.jsonl" --output "../data" --model deepseek-chat
