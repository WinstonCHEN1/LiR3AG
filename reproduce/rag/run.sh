python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --index_dir "../../index/HotpotQA" --model qwen3:8b --top_k 5 --think
python qwen3.py --input "../../query/2WikiMultiHopQA.jsonl" --output "../data" --index_dir "../../index/2WikiMultiHopQA" --model qwen3:8b --top_k 5 --think
python qwen3.py --input "../../query/MultiHopRAG.jsonl" --output "../data" --index_dir "../../index/MultiHopRAG" --model qwen3:8b --top_k 5 --think
python qwen3.py --input "../../query/MuSiQue.jsonl" --output "../data" --index_dir "../../index/MuSiQue" --model qwen3:8b --top_k 5 --think

python qwen3.py --input "../../query/HotpotQA.jsonl" --output "../data" --index_dir "../../index/HotpotQA" --model qwen3:8b --top_k 5 --nothink
python qwen3.py --input "../../query/2WikiMultiHopQA.jsonl" --output "../data" --index_dir "../../index/2WikiMultiHopQA" --model qwen3:8b --top_k 5 --nothink
python qwen3.py --input "../../query/MultiHopRAG.jsonl" --output "../data" --index_dir "../../index/MultiHopRAG" --model qwen3:8b --top_k 5 --nothink
python qwen3.py --input "../../query/MuSiQue.jsonl" --output "../data" --index_dir "../../index/MuSiQue" --model qwen3:8b --top_k 5 --nothink

