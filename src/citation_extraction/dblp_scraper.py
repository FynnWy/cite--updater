"""
Script to scrape paper lists from major AI conferences on DBLP for the past 10 years.
Uses the DBLP API to fetch conference proceedings in JSON format.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from tqdm import tqdm
import argparse
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Major AI conferences and their DBLP configuration
# Format: {conference_name: (dblp_code, file_prefix, format)}
# dblp_code: directory code used in DBLP URLs (e.g., 'iclr', 'nips')
# file_prefix: prefix used in filename (e.g., 'iclr', 'neurips') - can differ from dblp_code
# format: file format extension ('bht' or 'html')
MAJOR_AI_CONFERENCES = {
    'ICLR': ('iclr', 'iclr', 'bht'),           # International Conference on Learning Representations
    'NeurIPS': ('nips', 'neurips', 'bht'),    # Neural Information Processing Systems (uses 'neurips' in filename)
    'ICML': ('icml', 'icml', 'bht'),          # International Conference on Machine Learning
    'AAAI': ('aaai', 'aaai', 'bht'),          # Association for the Advancement of Artificial Intelligence
    'FACCT': ('fat', 'facct', 'bht'),         # Conference on Fairness, Accountability, and Transparency
    'IJCAI': ('ijcai', 'ijcai', 'bht'),       # International Joint Conference on Artificial Intelligence
    'CVPR': ('cvpr', 'cvpr', 'bht'),          # Computer Vision and Pattern Recognition
    'ICCV': ('iccv', 'iccv', 'bht'),          # International Conference on Computer Vision
    'ECCV': ('eccv', 'eccv', 'bht'),          # European Conference on Computer Vision
    'ACL': ('acl', 'acl', 'bht'),             # Association for Computational Linguistics
    'EMNLP': ('emnlp', 'emnlp', 'bht'),       # Empirical Methods in Natural Language Processing
    'NAACL': ('naacl', 'naacl', 'bht'),       # North American Chapter of the ACL
    'CoNLL': ('conll', 'conll', 'bht'),       # Conference on Computational Natural Language Learning
}

# DBLP API base URL
DBLP_API_BASE = "https://dblp.org/search/publ/api"


def fetch_dblp_papers(
    dblp_code: str,
    file_prefix: str,
    format_ext: str,
    year: int,
    hits_per_page: int = 1000,
    delay: float = 1.0
) -> Optional[Dict]:
    """
    Fetch papers from DBLP for a specific conference and year.
    
    Args:
        dblp_code: DBLP directory code (e.g., 'iclr', 'nips')
        file_prefix: File prefix used in filename (e.g., 'iclr', 'neurips')
        format_ext: File format extension ('bht' or 'html')
        year: Year to fetch papers from
        hits_per_page: Number of hits per page (default: 1000)
        delay: Delay between requests in seconds (to be respectful to the API)
        
    Returns:
        Dictionary containing the JSON response from DBLP, or None if request fails
    """
    # Construct the query URL
    # Format: toc:db/conf/{dblp_code}/{file_prefix}{year}.{format_ext}:
    query = f"toc:db/conf/{dblp_code}/{file_prefix}{year}.{format_ext}:"
    # URL encode the query parameter
    encoded_query = quote(query, safe='')
    url = f"{DBLP_API_BASE}?q={encoded_query}&h={hits_per_page}&format=json"
    
    try:
        logging.info(f"Fetching papers from {dblp_code.upper()} {year}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Add delay to be respectful to the API
        time.sleep(delay)
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching papers from {dblp_code.upper()} {year}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response for {dblp_code.upper()} {year}: {e}")
        return None


def fetch_all_papers_with_pagination(
    dblp_code: str,
    file_prefix: str,
    format_ext: str,
    year: int,
    hits_per_page: int = 1000,
    delay: float = 1.0
) -> List[Dict]:
    """
    Fetch all papers from DBLP, handling pagination if needed.
    
    Args:
        dblp_code: DBLP directory code (e.g., 'iclr', 'nips')
        file_prefix: File prefix used in filename (e.g., 'iclr', 'neurips')
        format_ext: File format extension ('bht' or 'html')
        year: Year to fetch papers from
        hits_per_page: Number of hits per page
        delay: Delay between requests in seconds
        
    Returns:
        List of all paper entries (from the 'hit' field)
    """
    all_papers = []
    start_index = 0
    
    while True:
        # Construct the query URL with pagination
        query = f"toc:db/conf/{dblp_code}/{file_prefix}{year}.{format_ext}:"
        # URL encode the query parameter
        encoded_query = quote(query, safe='')
        url = f"{DBLP_API_BASE}?q={encoded_query}&h={hits_per_page}&f={start_index}&format=json"
        
        try:
            logging.info(f"Fetching papers from {dblp_code.upper()} {year} (starting at {start_index})...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract hits from the response
            hits = data.get('result', {}).get('hits', {})
            papers = hits.get('hit', [])
            
            # Handle both single paper (dict) and multiple papers (list) cases
            if isinstance(papers, dict):
                papers = [papers]
            elif not isinstance(papers, list):
                papers = []
            
            if not papers:
                break
            
            all_papers.extend(papers)
            
            # Check if we've fetched all papers
            total_hits = int(hits.get('@total', 0))
            computed_hits = int(hits.get('@computed', 0))
            sent_hits = int(hits.get('@sent', 0))
            
            logging.info(f"Fetched {len(papers)} papers (total: {total_hits}, fetched so far: {len(all_papers)})")
            
            # If we've fetched all available papers, break
            if len(all_papers) >= total_hits or start_index + sent_hits >= total_hits:
                break
            
            # Move to next page
            start_index += sent_hits
            
            # Add delay between requests
            time.sleep(delay)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching papers from {dblp_code.upper()} {year} (start={start_index}): {e}")
            break
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON response for {dblp_code.upper()} {year}: {e}")
            break
        except KeyError as e:
            logging.error(f"Unexpected response structure for {dblp_code.upper()} {year}: {e}")
            break
    
    return all_papers


def scrape_conference_year(
    conference_name: str,
    dblp_code: str,
    file_prefix: str,
    format_ext: str,
    year: int,
    output_dir: Path,
    hits_per_page: int = 1000,
    delay: float = 1.0,
    use_pagination: bool = True
) -> bool:
    """
    Scrape papers for a specific conference and year, save to JSON file.
    
    Args:
        conference_name: Conference display name (e.g., 'ICLR', 'NeurIPS')
        dblp_code: DBLP directory code (e.g., 'iclr', 'nips')
        file_prefix: File prefix used in filename (e.g., 'iclr', 'neurips')
        format_ext: File format extension ('bht' or 'html')
        year: Year to scrape
        output_dir: Directory to save JSON files
        hits_per_page: Number of hits per page
        delay: Delay between requests
        use_pagination: Whether to use pagination to fetch all papers
        
    Returns:
        True if successful, False otherwise
    """
    # Create output directory structure: output_dir/conference/year/
    conference_dir = output_dir / conference_name.upper()
    conference_dir.mkdir(parents=True, exist_ok=True)
    
    if use_pagination:
        papers = fetch_all_papers_with_pagination(dblp_code, file_prefix, format_ext, year, hits_per_page, delay)
    else:
        data = fetch_dblp_papers(dblp_code, file_prefix, format_ext, year, hits_per_page, delay)
        if data is None:
            return False
        
        # Extract papers from response
        hits = data.get('result', {}).get('hits', {})
        papers = hits.get('hit', [])
        
        # Handle both single paper (dict) and multiple papers (list) cases
        if isinstance(papers, dict):
            papers = [papers]
        elif not isinstance(papers, list):
            papers = []
    
    if not papers:
        logging.warning(f"No papers found for {conference_name.upper()} {year}")
        return False
    
    # Save to JSON file
    output_file = conference_dir / f"{conference_name.upper()}_{year}.json"
    
    # Create a structured output with metadata
    output_data = {
        'conference': conference_name.upper(),
        'year': year,
        'total_papers': len(papers),
        'scraped_at': datetime.now().isoformat(),
        'papers': papers
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved {len(papers)} papers to {output_file}")
    return True


def scrape_all_conferences(
    years: List[int],
    conferences: Optional[Dict[str, Tuple[str, str, str]]] = None,
    output_dir: str = "data/dblp_conferences",
    hits_per_page: int = 1000,
    delay: float = 1.0,
    use_pagination: bool = True
) -> None:
    """
    Scrape papers from all specified conferences for the given years.
    
    Args:
        years: List of years to scrape
        conferences: Dictionary of conference names and (dblp_code, file_prefix, format_ext) tuples
                     (default: MAJOR_AI_CONFERENCES)
        output_dir: Directory to save JSON files
        hits_per_page: Number of hits per page
        delay: Delay between requests
        use_pagination: Whether to use pagination to fetch all papers
    """
    if conferences is None:
        conferences = MAJOR_AI_CONFERENCES
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        'total_conferences': 0,
        'total_years': 0,
        'successful': 0,
        'failed': 0
    }
    
    # Create progress bar
    total_tasks = len(conferences) * len(years)
    with tqdm(total=total_tasks, desc="Scraping conferences") as pbar:
        for conference_name, (dblp_code, file_prefix, format_ext) in conferences.items():
            for year in years:
                stats['total_conferences'] = len(conferences)
                stats['total_years'] = len(years)
                
                success = scrape_conference_year(
                    conference_name,
                    dblp_code,
                    file_prefix,
                    format_ext,
                    year,
                    output_path,
                    hits_per_page,
                    delay,
                    use_pagination
                )
                
                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                
                pbar.update(1)
    
    # Print summary statistics
    logging.info("\n" + "="*50)
    logging.info("Scraping Summary")
    logging.info("="*50)
    logging.info(f"Total conferences: {stats['total_conferences']}")
    logging.info(f"Years per conference: {stats['total_years']}")
    logging.info(f"Successful scrapes: {stats['successful']}")
    logging.info(f"Failed scrapes: {stats['failed']}")
    logging.info("="*50)


def main():
    """Main function to parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape paper lists from major AI conferences on DBLP'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2015,
        help='Start year for scraping (default: 2015, last 10 years)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help='End year for scraping (default: current year)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/dblp_conferences',
        help='Output directory for JSON files (default: data/dblp_conferences)'
    )
    parser.add_argument(
        '--hits-per-page',
        type=int,
        default=1000,
        help='Number of hits per page (default: 1000)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--conferences',
        type=str,
        nargs='+',
        default=None,
        help='Specific conferences to scrape (default: all major AI conferences)'
    )
    parser.add_argument(
        '--no-pagination',
        action='store_true',
        help='Disable pagination (only fetch first page)'
    )
    
    args = parser.parse_args()
    
    # Determine end year
    end_year = args.end_year if args.end_year else datetime.now().year
    years = list(range(args.start_year, end_year + 1))
    
    # Filter conferences if specified
    conferences = None
    if args.conferences:
        conferences = {}
        # Create a case-insensitive mapping
        conf_dict_lower = {k.lower(): k for k in MAJOR_AI_CONFERENCES.keys()}
        for conf_name in args.conferences:
            conf_name_lower = conf_name.lower()
            if conf_name_lower in conf_dict_lower:
                original_key = conf_dict_lower[conf_name_lower]
                conferences[original_key] = MAJOR_AI_CONFERENCES[original_key]
            else:
                logging.warning(f"Unknown conference: {conf_name}. Skipping.")
    
    logging.info(f"Scraping papers from {len(conferences or MAJOR_AI_CONFERENCES)} conferences")
    logging.info(f"Years: {args.start_year} to {end_year}")
    logging.info(f"Output directory: {args.output_dir}")
    
    scrape_all_conferences(
        years=years,
        conferences=conferences,
        output_dir=args.output_dir,
        hits_per_page=args.hits_per_page,
        delay=args.delay,
        use_pagination=not args.no_pagination
    )


if __name__ == '__main__':
    main()

