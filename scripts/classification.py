from openai import OpenAI
import json
import re
import os
client = OpenAI()
input_file = "./HotpotQA_rag_deepseek-r1-250120_20250718_14_36_24.jsonl"
output_file = "output.jsonl"

think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

classification_prompt_template = """
You need to classify the provided reasoning paragraphs into one of the following categories based on the following reasoning content description:
1. Use the provided context as evidence or clues for reasoning
2. Determine whether the context is relevant, select the relevant ones, remove the irrelevant ones, eliminate conflicts, and then generate results
3. Others
Please only return the corresponding numerical number (for example: 1, 2, 3) without outputting any other content.
Thinking content:
{}
"""

def classify_text(text):
    prompt = classification_prompt_template.format(text.strip())
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classification assistant, helping to determine whether the inference content belongs to one of several paradigms."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            match = think_pattern.search(data.get("generate", ""))
            if match:
                think_content = match.group(1)
                label = classify_text(think_content)
                data["label"] = label
            else:
                data["label"] = "N/A"
        except Exception as e:
            data = {"error": str(e)}
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        print("Finish")
