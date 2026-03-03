"""Example: quick DBLP parser lookup.

Requires a local DBLP XML (e.g., data/dblp.xml). Demonstrates a few title
queries and prints/serializes the results.

Usage:
    python examples/dblp_parser_demo.py --dblp-xml data/dblp.xml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from src.parser.dblp_parser import DblpParser

SAMPLE_TITLES: List[str] = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Deep Residual Learning for Image Recognition",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few sample lookups on a DBLP XML file.")
    parser.add_argument("--dblp-xml", type=Path, default=Path("data/dblp.xml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dblp_parser_sample.json")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    if not args.dblp_xml.exists():
        raise SystemExit(f"DBLP XML not found: {args.dblp_xml}")

    dblp = DblpParser(str(args.dblp_xml), cache_dir="dblp_cache")

    results: List[Dict] = []
    for title in SAMPLE_TITLES:
        hit = dblp.search_by_title(title, threshold=args.threshold)
        results.append({"query": title, "result": hit})
        print(f"{title[:50]:50s} -> {'found' if hit else 'not found'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sample results to {args.output}")


if __name__ == "__main__":
    main()
