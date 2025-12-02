import logging
import time
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def setup_logger(log_file_name):
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

def run_tokenization_experiment(inputs, outputs, model_name,
                                 full_response_objs=None,
                                 log_file_name="tokenizer_log.txt",
                                 total_duration=None):
    setup_logger(log_file_name)

    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    num_items = len(inputs)

    for idx, (inp, outp) in enumerate(zip(inputs, outputs), 1):
        in_tokens = tokenizer.encode(inp)
        out_tokens = tokenizer.encode(outp)
        total_input_tokens += len(in_tokens)
        total_output_tokens += len(out_tokens)

        if model_name in REASONING_MODELS:
            reasoning_text = extract_reasoning_text(
                outp,
                model_name,
                full_response_objs[idx - 1] if full_response_objs else None
            )
            reasoning_tokens = tokenizer.encode(reasoning_text)
        else:
            reasoning_tokens = []

        total_reasoning_tokens += len(reasoning_tokens)

        logging.info(f"[Query {idx}] Input Tokens: {len(in_tokens)}")
        logging.info(f"[Query {idx}] Output Tokens: {len(out_tokens)}")
        logging.info(f"[Query {idx}] Reasoning Tokens: {len(reasoning_tokens)}")

    avg_input = total_input_tokens / num_items if num_items else 0
    avg_output = total_output_tokens / num_items if num_items else 0
    avg_reasoning = total_reasoning_tokens / num_items if num_items else 0

    logging.info(f"Total: {num_items}")
    logging.info(f"Total Input Tokens: {total_input_tokens}")
    logging.info(f"Total Output Tokens: {total_output_tokens}")
    logging.info(f"Total Reasoning Tokens: {total_reasoning_tokens}")
    logging.info(f"Avg. Input Tokens: {avg_input:.2f}")
    logging.info(f"Avg. Output Tokens: {avg_output:.2f}")
    logging.info(f"Avg. Reasoning Tokens: {avg_reasoning:.2f}")
    logging.info(f"Total time:{total_duration:.4f} s" if total_duration else "No time")
