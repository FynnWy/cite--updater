"""Unified entry point for the two-stage reference-checking pipeline.

Stage 1: Citation extraction
  download  -> fetch PDFs from arXiv based on DBLP metadata
  grobid    -> run GROBID fulltext extraction on PDFs
  parse     -> turn GROBID XML into tabular metadata

Stage 2: Name matching against databases
  validate  -> match/validate citations against DBLP (and other sources)
  classify  -> optional LLM-based error classification
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Common default paths
PDF_DIR = BASE_DIR / "data" / "arxiv_pdfs"
GROBID_XML_DIR = BASE_DIR / "data" / "outputs" / "arxiv_pdfs"
METADATA_CSV = BASE_DIR / "data" / "arxiv_metadata.csv"
PARSED_JSON_DIR = BASE_DIR / "data" / "parsed_jsons"
DBLP_XML = BASE_DIR / "data" / "dblp.xml"
VALIDATION_DIR = BASE_DIR / "validation_results"
VALIDATION_JSON = VALIDATION_DIR / "validation_results.json"
CLASSIFIED_JSON = VALIDATION_DIR / "classified_results.json"


def run_subprocess(module: str, extra_args: list[str]) -> None:
    """Run a module as a subprocess with `python -m module ...`."""
    cmd = [sys.executable, "-m", module, *extra_args]
    subprocess.run(cmd, check=True)


def cmd_download(args: argparse.Namespace) -> None:
    from src.citation_extraction import arxiv_fetcher

    arxiv_fetcher.setup_logging(args.log_file)
    resume = not args.no_resume
    max_papers = args.max_papers if args.max_papers and args.max_papers > 0 else None
    arxiv_fetcher.process_all_conferences(
        output_dir=str(args.output_dir),
        max_papers=max_papers,
        match_threshold=args.match_threshold,
        delay=args.delay,
        resume=resume,
        log_file=args.log_file,
        metadata_file=str(args.metadata_file),
    )


def cmd_grobid(args: argparse.Namespace) -> None:
    from src.citation_extraction.grobid_runner import run_grobid

    run_grobid(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        config_path=str(args.config_path),
        verbose=not args.no_verbose,
        force=not args.no_force,
    )


def cmd_parse(args: argparse.Namespace) -> None:
    from src.citation_extraction.grobid_parser import process_all

    process_all(
        input_dir=Path(args.input_dir),
        pattern=args.pattern,
        output_csv=Path(args.output_csv),
    )


def cmd_validate(args: argparse.Namespace) -> None:
    cli_args = [
        "--input-dir",
        str(args.input_dir),
        "--dblp-xml",
        str(args.dblp_xml),
        "--output-dir",
        str(args.output_dir),
        "--threshold",
        str(args.threshold),
        "--title-similarity-threshold",
        str(args.title_similarity_threshold),
    ]
    if args.num_files:
        cli_args += ["--num-files", str(args.num_files)]
    run_subprocess("src.name_matching.validate_citations", cli_args)


def cmd_classify(args: argparse.Namespace) -> None:
    cli_args = [
        "--input_file",
        str(args.input_file),
        "--output_file",
        str(args.output_file),
    ]
    if args.max_samples:
        cli_args += ["--max_samples", str(args.max_samples)]
    if args.gpu_memory_utilization is not None:
        cli_args += ["--gpu_memory_utilization", str(args.gpu_memory_utilization)]
    run_subprocess("src.models.vllm_classifier", cli_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for the citation processing pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download PDFs from arXiv")
    p_dl.add_argument("--output-dir", type=Path, default=PDF_DIR)
    p_dl.add_argument("--max-papers", type=int, default=None)
    p_dl.add_argument("--match-threshold", type=int, default=85)
    p_dl.add_argument("--delay", type=float, default=3.0)
    p_dl.add_argument("--no-resume", action="store_true", help="start fresh, ignore checkpoints")
    p_dl.add_argument("--log-file", type=str, default="arxiv_download_progress.log")
    p_dl.add_argument("--metadata-file", type=Path, default=BASE_DIR / "data" / "arxiv_papers_metadata.json")
    p_dl.set_defaults(func=cmd_download)

    # grobid
    p_gr = sub.add_parser("grobid", help="Run GROBID on downloaded PDFs")
    p_gr.add_argument("--input-dir", type=Path, default=PDF_DIR)
    p_gr.add_argument("--output-dir", type=Path, default=GROBID_XML_DIR)
    p_gr.add_argument("--config-path", type=Path, default=BASE_DIR / "config" / "config.json")
    p_gr.add_argument("--no-verbose", action="store_true")
    p_gr.add_argument("--no-force", action="store_true")
    p_gr.set_defaults(func=cmd_grobid)

    # parse
    p_ps = sub.add_parser("parse", help="Parse GROBID TEI XML to CSV/TSV")
    p_ps.add_argument("--input-dir", type=Path, default=GROBID_XML_DIR)
    p_ps.add_argument("--pattern", type=str, default="*.grobid.tei.xml")
    p_ps.add_argument("--output-csv", type=Path, default=METADATA_CSV)
    p_ps.set_defaults(func=cmd_parse)

    # validate
    p_va = sub.add_parser("validate", help="Validate citations against DBLP")
    p_va.add_argument("--input-dir", type=Path, default=PARSED_JSON_DIR)
    p_va.add_argument("--dblp-xml", type=Path, default=DBLP_XML)
    p_va.add_argument("--output-dir", type=Path, default=VALIDATION_DIR)
    p_va.add_argument("--num-files", type=int, default=None)
    p_va.add_argument("--threshold", type=float, default=5.0)
    p_va.add_argument("--title-similarity-threshold", type=float, default=95.0)
    p_va.set_defaults(func=cmd_validate)

    # classify
    p_cl = sub.add_parser("classify", help="Classify mismatches with vLLM")
    p_cl.add_argument("--input_file", type=Path, default=VALIDATION_JSON)
    p_cl.add_argument("--output_file", type=Path, default=CLASSIFIED_JSON)
    p_cl.add_argument("--max_samples", type=int, default=None)
    p_cl.add_argument("--gpu_memory_utilization", type=float, default=None)
    p_cl.set_defaults(func=cmd_classify)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
