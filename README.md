# LiR3AG

- [[AAAI 2026] A Lightweight Rerank Reasoning Strategy Framework for Retrieval-Augmented Generation] (https://arxiv.org/abs/2512.18329)

## Environmental Installation

```bash
conda create -n lir3ag python=3.10
conda activate lir3ag
pip install -r requirements.txt
```
## Quick Start

### Datasets

In this paper, we use the following datasets for evaluation:
- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [MultiHopRAG](https://github.com/yixuantt/MultiHop-RAG)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)

### Prerequisites
- A prepared document index with **FAISS** (if your workflow uses vector retrieval).
- Input dataset in JSON or JSONL format (fields typically: `id`, `question` / `query`; corpus items: `id`, `text` / `content` / `passage`).

### Basic usage examples

Run the reproduce example that calls Ollama (example):
```bash
python LiR3AG/reproduce/rag/qwen3.py \
  --input "../../query/HotpotQA.jsonl" \
  --output "../data" \
  --index_dir "../../index/HotpotQA" \
  --model qwen3:14b \
  --top_k 5 \
  --think
```

### Output
- Outputs are saved as JSON / JSONL files.
- Typical output fields: `id`, `question`, `think` (intermediate chain-of-thought or reasoning), `response` (final answer), and optionally `retrieved_docs` or `ranked_passages`.

### Tips
- Ensure the index has been generated before running retrieval experiments.
- Start with a small subset of data to tune hyperparameters (e.g., `--top_k`, prompt templates, model selection).
- Use descriptive output directories (experiment name + timestamp) to keep results organized.

For more examples and parameter descriptions, inspect the scripts under LiR3AG/scripts and LiR3AG/reproduce.

## Citation
```
@article{chen2025lir,
  title={LIR $\^{} 3$ AG: A Lightweight Rerank Reasoning Strategy Framework for Retrieval-Augmented Generation},
  author={Chen, Guo and Huang, Junjie and Xie, Huaijin and Sun, Fei and Jia, Tao},
  journal={arXiv preprint arXiv:2512.18329},
  year={2025}
}
```
