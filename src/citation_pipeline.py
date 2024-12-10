"""
Citation Author Matcher Pipeline

This script processes academic papers from GROBID-parsed XML files and matches them against
arXiv and DBLP databases to verify and normalize author information.

Usage:
    python citation_pipeline.py [options]
    
Options:
    --dry-run              Run with a random sample of papers
    --sample N            Number of papers to sample in dry run (default: 5)
    --input FILE          Input XML file path
    --output FILE         Output JSON file path
    --threshold N         Title match threshold (default: 80)
    --dblp-delay N        Delay between DBLP API calls in seconds (default: 1)
    --arxiv-delay N       Delay between arXiv API calls in seconds (default: 0.5)
"""

import xml.etree.ElementTree as ET
import arxiv
import json
from nameparser import HumanName
from fuzzywuzzy import fuzz
import requests
from urllib.parse import urlencode
import time
from typing import Dict, List, Optional, Any
import backoff
import re
import random
import argparse
import os
from requests.exceptions import HTTPError
import logging
from parser.dblp_parser import DblpParser

logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citation_pipeline.log'),
        # Removed StreamHandler to prevent console output
    ]
)

def normalize_author_name(name: str) -> Dict[str, str]:
    """
    Normalize author names into a consistent format using nameparser.
    Removes both 4-digit suffixes and DBLP-style numeric suffixes.
    
    Args:
        name (str): Full author name string
        
    Returns:
        Dict[str, str]: Normalized name components
    """
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)  # Remove trailing numbers like 0001
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)  # Remove internal numbers
    
    # Parse the cleaned name
    parsed = HumanName(cleaned_name)
    
    return {
        'first_name': parsed.first or '',
        'middle_name': parsed.middle or '',
        'last_name': parsed.last or '',  # Last name should now be clean of numeric suffixes
        'suffix': parsed.suffix or '',
        'title': parsed.title or '',
        'original': name  # Keep the original name for reference
    }

def parse_xml(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse XML and return normalized author information."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        publications = {}
        
        for biblStruct in root.findall('.//tei:biblStruct', ns):
            title = biblStruct.find('.//tei:title[@type="main"]', ns)
            title_text = title.text if title is not None else "No title available"
            
            authors = []
            for author in biblStruct.findall('.//tei:author/tei:persName', ns):
                forename = author.find('tei:forename', ns)
                surname = author.find('tei:surname', ns)
                if forename is not None and surname is not None:
                    full_name = f"{forename.text} {surname.text}"
                    authors.append(normalize_author_name(full_name))
            
            publications[title_text] = authors
        return publications
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return {}

def query_arxiv(title: str, match_threshold: int, delay: float) -> Optional[List[Dict[str, str]]]:
    """Query arXiv and return normalized author information."""
    try:
        print(f"\nQuerying arXiv for: {title[:100]}...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=title,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search):
            title_similarity = fuzz.ratio(result.title.lower(), title.lower())
            print(f"arXiv match score: {title_similarity}")
            if title_similarity > match_threshold:
                return [normalize_author_name(author.name) for author in result.authors]
        return None
    except Exception as e:
        print(f"Error querying arXiv: {e}")
        return None
    finally:
        time.sleep(delay)

# Initialize the DBLP parser with explicit index path
dblp_parser = DblpParser(
    xml_path="dblp/dblp-2024-11-04.xml", 
    cache_dir="dblp_cache",
    index_name="dblp_index"  # This will be used as the index name
)

def query_dblp_with_parser(title: str, match_threshold: int) -> Optional[List[Dict[str, str]]]:
    """Query DBLP using the DblpParser."""
    try:
        logging.debug(f"Querying DBLP for: {title[:100]}...")
        result = dblp_parser.search_by_title(title)
        
        if result:
            # Compare titles using fuzzy matching
            dblp_title = result.get('title', '')
            title_similarity = fuzz.ratio(dblp_title.lower(), title.lower())
            logging.info(f"DBLP match score: {title_similarity}")
            
            if title_similarity > match_threshold:
                authors = result.get('authors', [])
                # Normalize author names
                return [normalize_author_name(author) for author in authors]
            else:
                logging.info(f"Title match score {title_similarity} below threshold {match_threshold}")
                return None
        return None
    except Exception as e:
        logging.error(f"Unexpected error querying DBLP: {e}")
        return None

def check_name_match(parsed_name: Dict[str, str], matched_name: Dict[str, str]) -> Optional[str]:
    """
    Check if two author names match according to specified rules.
    Returns mismatch description if names don't match, None otherwise.
    """
    # Compare first names
    if parsed_name['first_name'] and matched_name['first_name']:
        if parsed_name['first_name'] != matched_name['first_name']:
            return f"First name mismatch: {parsed_name['first_name']} vs {matched_name['first_name']}"
            
    # Compare last names
    if parsed_name['last_name'] != matched_name['last_name']:
        # Special case: handle numeric suffixes in last names (e.g., "0001")
        if not matched_name['last_name'].isdigit() and parsed_name['last_name'] != matched_name['last_name']:
            return f"Last name mismatch: {parsed_name['last_name']} vs {matched_name['last_name']}"
    
    # Compare middle initials if both exist
    if parsed_name['middle_name'] and matched_name['middle_name']:
        if parsed_name['middle_name'][0] != matched_name['middle_name'][0]:
            return f"Middle initial mismatch: {parsed_name['middle_name']} vs {matched_name['middle_name']}"
            
    return None

def process_publications(
    input_file: str,
    output_file: str,
    match_threshold: int,
    dblp_delay: float,
    arxiv_delay: float,
    dry_run: bool = False,
    sample_size: int = 5
) -> List[Dict]:
    """
    Main processing function that coordinates the entire pipeline.
    Currently only checking DBLP (arXiv temporarily disabled).
    """
    logging.info(f"\nProcessing publications from {input_file}")
    publications = parse_xml(input_file)
    if not publications:
        logging.warning("No publications found in XML file")
        return []
    
    if dry_run:
        titles = list(publications.keys())
        sample_size = min(sample_size, len(titles))
        sampled_titles = random.sample(titles, sample_size)
        publications = {title: publications[title] for title in sampled_titles}
        logging.info(f"\nDRY RUN: Processing {sample_size} random publications")
        
    logging.info(f"Found {len(publications)} publications to process")
    results = []
    matched_count = 0
    
    for i, (title, parsed_authors) in enumerate(publications.items(), 1):
        logging.info(f"\nProcessing publication {i}/{len(publications)}")
        logging.debug(f"Title: {title}")
        
        # Try DBLP only
        if dblp_authors := query_dblp_with_parser(title, match_threshold):
            mismatches = []
            for idx, (parsed, matched) in enumerate(zip(parsed_authors, dblp_authors)):
                if mismatch := check_name_match(parsed, matched):
                    mismatches.append(f"Author {idx + 1}: {mismatch}")
                    
            results.append({
                'title': title,
                'parsed_authors': parsed_authors,
                'matched_authors': dblp_authors,
                'source': 'dblp',
                'mismatches': mismatches if mismatches else []
            })
            matched_count += 1
            logging.info("Found match in DBLP")
        else:
            logging.info("No matches found in DBLP")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"\nResults have been written to {output_file}")
        logging.info(f"Found matches for {matched_count} out of {len(publications)} publications")
        logging.info(f"DBLP matches: {sum(1 for r in results if r['source'] == 'dblp')}")
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Match paper authors against arXiv and DBLP.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    parser.add_argument('--input', type=str, 
                       default='pdfs/2024.lrec-main.320.grobid.tei.xml',
                       help='Input XML file path')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    
    # Matching options
    parser.add_argument('--threshold', type=int, default=80,
                       help='Title match threshold (0-100)')
    
    # Rate limiting options
    parser.add_argument('--dblp-delay', type=float, default=1.0,
                       help='Delay between DBLP API calls in seconds')
    parser.add_argument('--arxiv-delay', type=float, default=0.5,
                       help='Delay between arXiv API calls in seconds')
    
    # Dry run options
    parser.add_argument('--dry-run', action='store_true',
                       help='Run with a random sample of papers')
    parser.add_argument('--sample', type=int, default=5,
                       help='Number of papers to sample in dry run')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        base_name = 'dry_run_matches.json' if args.dry_run else 'author_matches.json'
        args.output = base_name
    
    # Validate arguments
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    if args.threshold < 0 or args.threshold > 100:
        parser.error("Threshold must be between 0 and 100")
    
    if args.dblp_delay < 0 or args.arxiv_delay < 0:
        parser.error("Delays must be non-negative")
    
    if args.sample < 1:
        parser.error("Sample size must be positive")
    
    # Run the pipeline
    results = process_publications(
        input_file=args.input,
        output_file=args.output,
        match_threshold=args.threshold,
        dblp_delay=args.dblp_delay,
        arxiv_delay=args.arxiv_delay,
        dry_run=args.dry_run,
        sample_size=args.sample
    )