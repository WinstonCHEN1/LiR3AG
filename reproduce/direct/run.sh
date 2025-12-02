python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --model qwen3:8b --nothink
python qwen3.py --input "../../query/2WikiMultiHopQA.jsonl" --output "../data" --model qwen3:8b --nothink
python qwen3.py --input "../../query/MultiHopRAG.jsonl" --output "../data" --model qwen3:8b --nothink
python qwen3.py --input "../../query/MuSiQue.jsonl" --output "../data" --model qwen3:8b --nothink

python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --model qwen3:8b --think
python qwen3.py --input "../../query/2WikiMultiHopQA.jsonl" --output "../data" --model qwen3:8b --think
python qwen3.py --input "../../query/MultiHopRAG.jsonl" --output "../data" --model qwen3:8b --think
python qwen3.py --input "../../query/MuSiQue.jsonl" --output "../data" --model qwen3:8b --think