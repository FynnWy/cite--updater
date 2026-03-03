"""
Script to download PDFs from arXiv using DBLP conference data.

This script loads paper titles from DBLP JSON files and queries arXiv API
to find matching papers. If a good match is found, it downloads the PDF.

Features:
- Processes all conferences automatically
- Resume capability from interruptions
- Progress logging and tracking
- Configurable matching thresholds

Usage:
    python -m src.citation_extraction.arxiv_fetcher [options]

Options:
    --output-dir DIR          Directory to save PDFs (default: data/arxiv_pdfs)
    --max-papers N            Maximum number of papers to process per conference (default: None for all)
    --match-threshold N       Title similarity threshold for arXiv matches (default: 85)
    --delay N                 Delay between arXiv API calls in seconds (default: 3)
    --resume                  Resume from last checkpoint (default: True)
    --log-file FILE          Progress log file (default: arxiv_download_progress.log)
"""

import json
import os
import time
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
from fuzzywuzzy import fuzz
import arxiv
from tqdm import tqdm
from datetime import datetime
from nameparser import HumanName

def clean_author_name(name: str) -> str:
    """
    Clean author names by removing four-digit suffixes and other numeric patterns.

    Args:
        name (str): Raw author name

    Returns:
        str: Cleaned author name
    """
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)  # Remove trailing numbers like 0001
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)  # Remove internal numbers

    # Parse the cleaned name and reconstruct it
    parsed = HumanName(cleaned_name.strip())
    # Return the name in "First Last" format, or the original cleaned name if parsing fails
    if parsed.first and parsed.last:
        return f"{parsed.first} {parsed.last}".strip()
    else:
        return cleaned_name.strip()

def load_metadata(metadata_file: str) -> Dict[str, Any]:
    """
    Load existing metadata from file.

    Args:
        metadata_file: Path to metadata file

    Returns:
        Dictionary containing metadata
    """
    if Path(metadata_file).exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load metadata from {metadata_file}: {e}")
            return {}
    return {}

def save_metadata(metadata_file: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata to file.

    Args:
        metadata_file: Path to save metadata
        metadata: Metadata dictionary
    """
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logging.error(f"Could not save metadata to {metadata_file}: {e}")

def setup_logging(log_file: str = "arxiv_download_progress.log"):
    """Configure logging with file and console output."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_all_conferences() -> List[Tuple[str, int]]:
    """
    Get list of all available conferences and years.

    Returns:
        List of (conference, year) tuples
    """
    conferences_dir = Path("data/dblp_conferences")
    conferences = []

    if not conferences_dir.exists():
        logging.error(f"Conferences directory not found: {conferences_dir}")
        return []

    for conf_dir in conferences_dir.iterdir():
        if conf_dir.is_dir():
            conference = conf_dir.name
            for json_file in conf_dir.glob("*.json"):
                # Extract year from filename (e.g., AAAI_2024.json -> 2024)
                try:
                    year = int(json_file.stem.split('_')[1])
                    conferences.append((conference, year))
                except (IndexError, ValueError):
                    logging.warning(f"Could not parse year from filename: {json_file.name}")
                    continue

    # Sort by conference then year
    conferences.sort(key=lambda x: (x[0], x[1]))
    return conferences

def load_conference_data(conference: str, year: int) -> Optional[Dict[str, Any]]:
    """
    Load conference data from JSON file.

    Args:
        conference: Conference name (e.g., 'AAAI')
        year: Conference year

    Returns:
        Conference data dictionary or None if file not found
    """
    json_path = Path(f"data/dblp_conferences/{conference}/{conference}_{year}.json")

    if not json_path.exists():
        logging.error(f"Conference file not found: {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {data['total_papers']} papers from {conference} {year}")
        return data
    except Exception as e:
        logging.error(f"Error loading conference data: {e}")
        return None

def load_progress(log_file: str) -> Dict[str, Any]:
    """
    Load progress from log file to enable resume functionality.

    Args:
        log_file: Path to progress log file

    Returns:
        Dictionary with progress information
    """
    progress = {
        'conferences_processed': set(),
        'papers_processed': {},
        'total_downloads': 0,
        'last_conference': None,
        'last_paper_idx': -1
    }

    if not Path(log_file).exists():
        return progress

    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Processing conference:' in line:
                    # Extract conference-year from log line
                    try:
                        parts = line.split('Processing conference:')[1].strip().split()
                        conf_year = f"{parts[0]}_{parts[1]}"
                        progress['last_conference'] = conf_year
                    except:
                        continue
                elif 'Processing paper' in line and '/' in line:
                    # Extract paper index from "Processing paper X/Y"
                    try:
                        paper_part = line.split('Processing paper')[1].split()[0]
                        current_paper = int(paper_part.split('/')[0])
                        progress['last_paper_idx'] = current_paper - 1  # Convert to 0-based index
                    except:
                        continue
                elif 'Downloaded PDF to' in line:
                    progress['total_downloads'] += 1
    except Exception as e:
        logging.warning(f"Could not load progress from {log_file}: {e}")

    return progress

def query_arxiv_by_title(title: str, match_threshold: int = 85) -> Optional[Dict[str, Any]]:
    """
    Query arXiv for a paper by title and return match info if found.

    Args:
        title: Paper title to search for
        match_threshold: Minimum title similarity score (0-100)

    Returns:
        Dictionary with arXiv paper info if match found, None otherwise
    """
    try:
        # Configure client to handle redirects properly (fixes HTTP 301 issues)
        client = arxiv.Client(
            page_size=5,
            delay_seconds=3.0,
            num_retries=3
        )
        search = arxiv.Search(
            query=title,
            max_results=5,  # Get a few results to find the best match
            sort_by=arxiv.SortCriterion.Relevance
        )

        best_match = None
        best_score = 0

        for result in client.results(search):
            # Calculate title similarity
            score = fuzz.ratio(result.title.lower(), title.lower())
            logging.debug(f"arXiv result: '{result.title[:100]}...' (score: {score})")

            if score > best_score:
                best_score = score
                best_match = result

        if best_match and best_score >= match_threshold:
            logging.info(f"Found arXiv match: '{best_match.title[:100]}...' (score: {best_score})")
            return {
                'arxiv_id': best_match.get_short_id(),
                'title': best_match.title,
                'authors': [author.name for author in best_match.authors],
                'pdf_url': best_match.pdf_url,
                'match_score': best_score
            }

        logging.info(f"No good arXiv match found (best score: {best_score})")
        return None

    except Exception as e:
        logging.error(f"Error querying arXiv: {e}")
        return None

def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download a PDF from a URL.

    Args:
        url: PDF URL
        output_path: Path to save the PDF

    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Downloaded PDF to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error downloading PDF: {e}")
        return False

def process_papers(conference_data: Dict[str, Any],
                  output_dir: str,
                  max_papers: Optional[int],
                  match_threshold: int,
                  delay: float,
                  start_paper_idx: int = 0,
                  metadata_file: str = None) -> int:
    """
    Process papers from conference data and download PDFs from arXiv.

    Args:
        conference_data: Conference data dictionary
        output_dir: Directory to save PDFs
        max_papers: Maximum number of papers to process (None for all)
        match_threshold: Title similarity threshold
        delay: Delay between API calls in seconds
        start_paper_idx: Paper index to start from (for resume)
        metadata_file: Path to metadata file (optional)

    Returns:
        Number of successful downloads
    """
    conference = conference_data['conference']
    year = conference_data['year']
    papers = conference_data['papers']

    # Create conference-specific directory
    conference_dir = Path(output_dir) / conference.lower() / str(year)
    conference_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if provided
    metadata = {}
    if metadata_file:
        metadata = load_metadata(metadata_file)

    # Determine how many papers to process
    total_papers = len(papers)
    if max_papers is None:
        max_papers = total_papers
    else:
        max_papers = min(max_papers, total_papers)

    logging.info(f"Processing papers {start_paper_idx + 1}-{max_papers} from {conference} {year} ({total_papers} total)")

    successful_downloads = 0
    processed_count = start_paper_idx

    # Process papers with progress bar
    papers_to_process = papers[start_paper_idx:max_papers]

    with tqdm(total=len(papers_to_process), desc=f"{conference} {year}", unit="paper") as pbar:
        for paper in papers_to_process:
            processed_count += 1

            # Get paper info
            paper_info = paper.get('info', {})
            title = paper_info.get('title', '').strip()

            if not title:
                logging.warning(f"Paper {processed_count}/{max_papers} has no title, skipping")
                pbar.update(1)
                continue

            logging.info(f"Processing paper {processed_count}/{max_papers}: '{title[:80]}...'")

            # Query arXiv
            arxiv_match = query_arxiv_by_title(title, match_threshold)

            if arxiv_match:
                # Create filename from arXiv ID
                arxiv_id = arxiv_match['arxiv_id']
                filename = f"{arxiv_id.replace('/', '_')}.pdf"
                pdf_path = conference_dir / filename

                # Check if already downloaded
                if pdf_path.exists():
                    logging.info(f"PDF already exists: {pdf_path}")
                    successful_downloads += 1

                    # Create metadata entry for existing file if not already in metadata
                    if metadata_file and arxiv_id not in metadata:
                        # Clean author names
                        cleaned_authors = [clean_author_name(author) for author in arxiv_match['authors']]

                        # Create metadata entry
                        metadata_entry = {
                            'arxiv_id': arxiv_id,
                            'title': arxiv_match['title'],
                            'authors': cleaned_authors,  # List of individual cleaned author names
                            'conference': conference,
                            'year': year,
                            'file_path': str(pdf_path.relative_to(Path(output_dir))),
                            'download_date': datetime.now().isoformat(),
                            'match_score': arxiv_match['match_score']
                        }

                        # Add to metadata using arXiv ID as key
                        metadata[arxiv_id] = metadata_entry
                        logging.debug(f"Added metadata for existing file {arxiv_id}")
                else:
                    # Download PDF
                    if download_pdf(arxiv_match['pdf_url'], pdf_path):
                        successful_downloads += 1
                        logging.info(f"Downloaded: {filename}")

                        # Create metadata entry
                        if metadata_file:
                            # Clean author names
                            cleaned_authors = [clean_author_name(author) for author in arxiv_match['authors']]

                            # Create metadata entry
                            metadata_entry = {
                                'arxiv_id': arxiv_id,
                                'title': arxiv_match['title'],
                                'authors': cleaned_authors,  # List of individual cleaned author names
                                'conference': conference,
                                'year': year,
                                'file_path': str(pdf_path.relative_to(Path(output_dir))),
                                'download_date': datetime.now().isoformat(),
                                'match_score': arxiv_match['match_score']
                            }

                            # Add to metadata using arXiv ID as key
                            metadata[arxiv_id] = metadata_entry
                            logging.debug(f"Added metadata for {arxiv_id}")
                    else:
                        logging.error(f"Failed to download PDF for arXiv ID: {arxiv_id}")
            else:
                logging.info("No arXiv match found for this paper")
            # Respect the delay between API calls
            if processed_count < max_papers:
                time.sleep(delay)

            pbar.update(1)

    logging.info(f"Conference {conference} {year} complete. Downloaded {successful_downloads} out of {len(papers_to_process)} papers to {conference_dir}.")

    # Save metadata if provided
    if metadata_file:
        save_metadata(metadata_file, metadata)

    return successful_downloads

def process_all_conferences(output_dir: str,
                          max_papers: Optional[int],
                          match_threshold: int,
                          delay: float,
                          resume: bool,
                          log_file: str,
                          metadata_file: str = None) -> Dict[str, Any]:
    """
    Process all available conferences and download PDFs.

    Args:
        output_dir: Directory to save PDFs
        max_papers: Maximum papers per conference
        match_threshold: Title similarity threshold
        delay: Delay between API calls
        resume: Whether to resume from checkpoint
        log_file: Progress log file
        metadata_file: Path to metadata file (optional)

    Returns:
        Summary statistics
    """
    # Get all conferences
    all_conferences = get_all_conferences()
    if not all_conferences:
        logging.error("No conferences found!")
        return {}

    # Load progress if resuming
    progress = load_progress(log_file) if resume else {}
    conferences_processed = progress.get('conferences_processed', set())
    last_conference = progress.get('last_conference', None)
    last_paper_idx = progress.get('last_paper_idx', -1)

    logging.info(f"Found {len(all_conferences)} conference-year combinations")
    logging.info(f"Resume mode: {'enabled' if resume else 'disabled'}")

    if resume and last_conference:
        logging.info(f"Resuming from {last_conference}, paper {last_paper_idx + 1}")

    total_downloads = progress.get('total_downloads', 0)
    conference_stats = []

    # Filter conferences to process
    conferences_to_process = []
    resume_found = False

    for conference, year in all_conferences:
        conf_year_key = f"{conference}_{year}"

        if resume and not resume_found:
            if conf_year_key == last_conference:
                resume_found = True
            else:
                continue  # Skip already processed conferences

        if conf_year_key not in conferences_processed or not resume:
            conferences_to_process.append((conference, year))

    logging.info(f"Will process {len(conferences_to_process)} conferences")

    # Process each conference
    for conference, year in tqdm(conferences_to_process, desc="Conferences", unit="conf"):
        conf_year_key = f"{conference}_{year}"

        logging.info(f"Processing conference: {conference} {year}")

        # Load conference data
        conference_data = load_conference_data(conference, year)
        if not conference_data:
            logging.error(f"Skipping {conference} {year} - could not load data")
            continue

        # Determine start paper index for resume
        start_idx = 0
        if resume and conf_year_key == last_conference:
            start_idx = max(0, last_paper_idx)  # Start from where we left off
            logging.info(f"Resuming {conference} {year} from paper {start_idx + 1}")

        # Process papers
        downloads = process_papers(
            conference_data=conference_data,
            output_dir=output_dir,
            max_papers=max_papers,
            match_threshold=match_threshold,
            delay=delay,
            start_paper_idx=start_idx,
            metadata_file=metadata_file
        )

        total_downloads += downloads
        conferences_processed.add(conf_year_key)

        # Save progress checkpoint
        conference_stats.append({
            'conference': conference,
            'year': year,
            'papers_total': len(conference_data['papers']),
            'papers_processed': min(max_papers or len(conference_data['papers']), len(conference_data['papers'])),
            'downloads': downloads
        })

        logging.info(f"Total downloads so far: {total_downloads}")

    # Final summary
    summary = {
        'total_conferences': len(conference_stats),
        'total_papers_processed': sum(stat['papers_processed'] for stat in conference_stats),
        'total_downloads': total_downloads,
        'conference_stats': conference_stats,
        'completed_at': datetime.now().isoformat()
    }

    logging.info("=== FINAL SUMMARY ===")
    logging.info(f"Conferences processed: {summary['total_conferences']}")
    logging.info(f"Papers processed: {summary['total_papers_processed']}")
    logging.info(f"PDFs downloaded: {summary['total_downloads']}")
    logging.info(f"Success rate: {summary['total_downloads']/max(summary['total_papers_processed'], 1)*100:.1f}%")

    # Save summary to file
    summary_file = Path(log_file).parent / "arxiv_download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Summary saved to {summary_file}")

    return summary

def main():
    """Main function to parse arguments and run the download process."""
    parser = argparse.ArgumentParser(
        description='Download PDFs from arXiv using DBLP conference data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output-dir', type=str, default='data/arxiv_pdfs',
                       help='Directory to save PDFs')
    parser.add_argument('--max-papers', type=int, default=None,
                       help='Maximum number of papers to process per conference (None for all)')
    parser.add_argument('--match-threshold', type=int, default=85,
                       help='Title similarity threshold for arXiv matches (0-100)')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Delay between arXiv API calls in seconds')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from last checkpoint')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoint (start fresh)')
    parser.add_argument('--log-file', type=str, default='arxiv_download_progress.log',
                       help='Progress log file')
    parser.add_argument('--metadata-file', type=str, default='data/arxiv_papers_metadata.json',
                       help='Metadata file for downloaded papers')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Handle resume logic
    resume = args.resume and not args.no_resume

    # Set max_papers to None if not specified
    max_papers = args.max_papers if args.max_papers and args.max_papers > 0 else None

    logging.info("=== arXiv PDF Downloader Started ===")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Max papers per conference: {max_papers or 'all'}")
    logging.info(f"Match threshold: {args.match_threshold}%")
    logging.info(f"Delay between requests: {args.delay}s")
    logging.info(f"Resume mode: {'enabled' if resume else 'disabled'}")

    # Process all conferences
    summary = process_all_conferences(
        output_dir=args.output_dir,
        max_papers=max_papers,
        match_threshold=args.match_threshold,
        delay=args.delay,
        resume=resume,
        log_file=args.log_file,
        metadata_file=args.metadata_file
    )

    print("\n=== PROCESSING COMPLETE ===")
    print(f"Conferences processed: {summary.get('total_conferences', 0)}")
    print(f"Papers processed: {summary.get('total_papers_processed', 0)}")
    print(f"PDFs downloaded: {summary.get('total_downloads', 0)}")

    if summary.get('total_papers_processed', 0) > 0:
        success_rate = summary['total_downloads'] / summary['total_papers_processed'] * 100
        print(f"Success rate: {success_rate:.1f}%")

    print(f"Log file: {args.log_file}")
    print(f"Summary file: arxiv_download_summary.json")
    print(f"Metadata file: {args.metadata_file}")

if __name__ == '__main__':
    main()
