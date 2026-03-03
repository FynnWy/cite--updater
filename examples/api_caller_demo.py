"""Example: multi-database paper search with the API caller.

Runs a few well-known titles through DBLP/arXiv/Semantic Scholar and writes
sample outputs next to this file.

Usage:
    python examples/api_caller_demo.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.name_matching.api_caller import (  # noqa: E402
    calculate_title_similarity,
    get_semantic_scholar_api_key,
    save_results_to_json,
    search_multiple_titles,
    search_papers_by_title,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_TITLES: List[str] = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Deep Residual Learning for Image Recognition",
]


def log_api_key_status() -> None:
    api_key = get_semantic_scholar_api_key()
    if api_key:
        print("Semantic Scholar API key loaded (higher rate limits enabled).")
    else:
        print("No Semantic Scholar API key found; using public rate limits.")


def log_title_similarity_examples() -> None:
    pairs = [
        ("Attention Is All You Need", "Attention is all you need"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "BERT Pretraining Deep Bidirectional Transformers"),
        ("Machine Learning", "Deep Learning"),
    ]
    for a, b in pairs:
        score = calculate_title_similarity(a, b)
        print(f"Similarity('{a}' vs '{b}') = {score}%")


def main() -> None:
    log_api_key_status()
    log_title_similarity_examples()

    # Single-title demo
    start = time.time()
    single = search_papers_by_title(SAMPLE_TITLES[0], similarity_threshold=80, max_results_per_source=5, parallel=True)
    print(f"\nSingle-title search finished in {time.time() - start:.2f}s")

    single_out = Path(__file__).parent / "api_caller_sample.json"
    save_results_to_json(single, single_out)
    print(f"Saved single-search sample to {single_out}")

    # Small batch demo
    start = time.time()
    batch = search_multiple_titles(
        SAMPLE_TITLES[:2], similarity_threshold=80, max_results_per_source=5, max_workers=2
    )
    print(f"Batch search (2 titles) finished in {time.time() - start:.2f}s")

    batch_out = Path(__file__).parent / "api_caller_batch_sample.json"
    save_results_to_json(batch, batch_out)
    print(f"Saved batch sample to {batch_out}")


if __name__ == "__main__":
    main()
