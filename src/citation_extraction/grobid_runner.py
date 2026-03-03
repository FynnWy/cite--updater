"""Thin wrapper around grobid_client to process PDF batches."""
import argparse
import logging
from pathlib import Path
from typing import Optional

from grobid_client.grobid_client import GrobidClient

logger = logging.getLogger(__name__)


def run_grobid(
    input_dir: str = "data/arxiv_pdfs",
    output_dir: str = "data/outputs/arxiv_pdfs",
    config_path: str = "./config/config.json",
    verbose: bool = True,
    force: bool = True,
) -> None:
    """Process PDFs through GROBID fulltext service."""
    client = GrobidClient(config_path=config_path)
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Starting GROBID: %s -> %s", in_path, out_path)
    client.process(
        "processFulltextDocument",
        str(in_path),
        output=str(out_path),
        verbose=verbose,
        json_output=True,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GROBID fulltext on a PDF tree")
    parser.add_argument("--input-dir", default="data/arxiv_pdfs")
    parser.add_argument("--output-dir", default="data/outputs/arxiv_pdfs")
    parser.add_argument("--config-path", default="./config/config.json")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    args = parser.parse_args()

    run_grobid(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        verbose=not args.no_verbose,
        force=not args.no_force,
    )


if __name__ == "__main__":
    main()
