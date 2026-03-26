"""
Download PDFs from ACL Anthology, optionally skipping titles available on arXiv.

This module restores the former ACL downloader workflow and adds an arXiv
availability check so ACL can be used as a complementary source.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import arxiv
import requests
from fuzzywuzzy import fuzz
from tqdm import tqdm

from src.utils import download_pdf


def setup_logging() -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def query_arxiv_by_title(title: str, match_threshold: int = 90) -> bool:
    """
    Return True if a strong title match exists on arXiv.

    This mirrors the arXiv matching style used by the existing arXiv downloader.
    """
    try:
        client = arxiv.Client(page_size=5, delay_seconds=0, num_retries=3)
        search = arxiv.Search(
            query=title,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        best_score = 0
        for result in client.results(search):
            score = fuzz.ratio(result.title.lower(), title.lower())
            if score > best_score:
                best_score = score

        return best_score >= match_threshold
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        logging.warning("arXiv availability check failed for '%s': %s", title[:80], exc)
        return False


def get_papers_by_year(anthology: object, year: int) -> List[object]:
    """
    Get all papers published in a specific year from ACL Anthology.

    Supports both modern and legacy ACL ID formats.
    """
    papers: List[object] = []
    year_str = str(year)
    year_short = year_str[-2:]
    old_format_pattern = re.compile(rf"^[A-Za-z]\d*{year_short}[-\.]\d+")

    for paper in anthology.papers():
        if paper.full_id.startswith(year_str):
            papers.append(paper)
        elif old_format_pattern.match(paper.full_id):
            papers.append(paper)

    return papers


def download_papers_by_year_range(
    start_year: int,
    end_year: Optional[int] = None,
    output_dir: str = "data/acl_pdfs",
    delay: float = 1.0,
    max_papers: Optional[int] = None,
    max_workers: int = 5,
    skip_if_on_arxiv: bool = True,
    arxiv_match_threshold: int = 90,
    arxiv_delay: float = 3.0,
) -> dict:
    """
    Download ACL papers by year range, optionally skipping titles found on arXiv.
    """
    if end_year is None:
        end_year = datetime.now().year

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from acl_anthology import Anthology
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "Missing dependency 'acl-anthology'. Install with: "
            "pip install -r requirements.txt -r requirements-acl.txt"
        ) from exc

    logging.info("Initializing ACL Anthology...")
    anthology = Anthology.from_repo()

    papers_downloaded = 0
    skipped_on_arxiv = 0
    skipped_existing = 0
    skipped_no_pdf = 0
    failed_downloads = 0

    def download_single_paper(paper: object, year_dir: Path) -> tuple[bool, str]:
        pdf_obj = getattr(paper, "pdf", None)
        if not pdf_obj:
            return False, "no_pdf"

        pdf_url = pdf_obj.url
        output_path = year_dir / f"{paper.full_id}.pdf"

        if output_path.exists():
            return False, "exists"

        ok = download_pdf(pdf_url, output_path)
        return (ok, "ok" if ok else "download_failed")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for year in range(start_year, end_year + 1):
            if max_papers is not None and papers_downloaded >= max_papers:
                break

            logging.info("Processing ACL Anthology year %s", year)
            year_papers = get_papers_by_year(anthology, year)
            if not year_papers:
                logging.info("No papers found for year %s", year)
                continue

            year_dir = output_root / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            selected_papers: List[object] = []
            for paper in tqdm(year_papers, desc=f"Filter {year}", unit="paper"):
                if max_papers is not None and (papers_downloaded + len(selected_papers)) >= max_papers:
                    break

                if not getattr(paper, "pdf", None):
                    skipped_no_pdf += 1
                    continue

                out_file = year_dir / f"{paper.full_id}.pdf"
                if out_file.exists():
                    skipped_existing += 1
                    continue

                if skip_if_on_arxiv:
                    title = getattr(paper, "title", "") or ""
                    if title and query_arxiv_by_title(title, match_threshold=arxiv_match_threshold):
                        skipped_on_arxiv += 1
                        time.sleep(arxiv_delay)
                        continue
                    time.sleep(arxiv_delay)

                selected_papers.append(paper)

            if not selected_papers:
                logging.info("No ACL-only papers selected for %s", year)
                continue

            futures = [executor.submit(download_single_paper, paper, year_dir) for paper in selected_papers]

            with tqdm(total=len(futures), desc=f"Download {year}", unit="paper") as pbar:
                for future in as_completed(futures):
                    ok, state = future.result()
                    if ok:
                        papers_downloaded += 1
                    elif state == "exists":
                        skipped_existing += 1
                    elif state == "no_pdf":
                        skipped_no_pdf += 1
                    else:
                        failed_downloads += 1
                    pbar.update(1)
                    if delay > 0:
                        time.sleep(delay / max(1, max_workers))

    summary = {
        "start_year": start_year,
        "end_year": end_year,
        "output_dir": str(output_root),
        "papers_downloaded": papers_downloaded,
        "skipped_on_arxiv": skipped_on_arxiv,
        "skipped_existing": skipped_existing,
        "skipped_no_pdf": skipped_no_pdf,
        "failed_downloads": failed_downloads,
        "completed_at": datetime.now().isoformat(),
    }
    logging.info("ACL downloader summary: %s", summary)
    return summary


def main() -> None:
    """CLI entrypoint for standalone ACL download runs."""
    parser = argparse.ArgumentParser(
        description="Download ACL Anthology PDFs, optionally skipping papers available on arXiv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=datetime.now().year - 10,
        help="Start year for paper downloads",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for paper downloads (default: current year)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/acl_pdfs",
        help="Directory where PDFs should be saved",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between ACL downloads in seconds",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to download",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel downloads",
    )
    parser.add_argument(
        "--include-arxiv",
        action="store_true",
        help="Do not skip titles that are also available on arXiv",
    )
    parser.add_argument(
        "--arxiv-match-threshold",
        type=int,
        default=90,
        help="Title similarity threshold for arXiv availability checks",
    )
    parser.add_argument(
        "--arxiv-delay",
        type=float,
        default=3.0,
        help="Delay between arXiv availability checks in seconds",
    )

    args = parser.parse_args()
    setup_logging()

    summary = download_papers_by_year_range(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        delay=args.delay,
        max_papers=args.max_papers,
        max_workers=args.max_workers,
        skip_if_on_arxiv=not args.include_arxiv,
        arxiv_match_threshold=args.arxiv_match_threshold,
        arxiv_delay=args.arxiv_delay,
    )
    print("ACL download complete:", summary)


if __name__ == "__main__":
    main()
