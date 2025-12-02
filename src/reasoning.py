import requests
from transformers import AutoTokenizer

def generate_reasoning_steps(question, context_list, model, client, api_url="http://localhost:11434/api/generate"):
    """Generate numbered reasoning steps (Chain of Thought) as a string."""
    indexed_contexts = "\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context_list)])
    prompt_template = '''
You are a helpful reasoning assistant. Given a **Question** and several **Contexts**, think step-by-step.

* If one or more contexts provide facts you need, you can follow them.
* If context is not useful, do not refer to it and provide steps of reasoning **based on your existing knowledge**.
* Write a concise, numbered chain-of-thought. Begin every line with "Step X:" (starting from 1).
* **Output only the reasoning steps.**

### Example
Contexts
1. Scott Derrickson is an American film director.
2. Bananas are rich in potassium.
3. Ed Wood (1924-1978) was an American filmmaker.

Question: Were Scott Derrickson and Ed Wood of the same nationality?

Reasoning (desired output):
Step 1: We know Scott Derrickson is American.
Step 2: We know Ed Wood is American.
Step 3: Since both are American, they share the same nationality.

---
{indexed_contexts}
Question: {question}
Write your reasoning steps now. Remember: only the steps, nothing else./no_think
'''
    prompt = prompt_template.format(indexed_contexts=indexed_contexts, question=question)
    
    # Tokenize the prompt for debugging
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(prompt)
    in_tokens = tokenizer.encode(prompt)
    print(len(in_tokens))
    import pdb; pdb.set_trace()

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
        try:
            response = requests.post(api_url, json={
                "model": "qwen3:8b",
                "prompt": prompt,
                "temperature": 0,
                "stream": False
            }, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error - Reasoning Generation] {e}")
            return ""