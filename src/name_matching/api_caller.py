"""
API Caller for Multi-Database Paper Search

This module provides functionality to search for papers across multiple databases
(DBLP, arXiv, Semantic Scholar) with proper rate limiting, parallelization, and
Open-Access filtering. Results are compared using fuzzy string matching to identify
likely matches.

Rate Limits:
- DBLP: 1 request per 3 seconds
- arXiv: 1 request per 3 seconds
- Semantic Scholar: 1 request per second (5 requests per 5 seconds with API key)

API Key Configuration:
To use Semantic Scholar API with higher rate limits, set your API key:
1. Create a .env file in the project root (copy from .env.example)
2. Add: SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
3. Get your API key from: https://www.semanticscholar.org/product/api

The .env file is automatically ignored by git to prevent accidental commits.

Usage:
    from api_caller import search_papers_by_title
    
    results = search_papers_by_title("Machine Learning in Computer Vision")

Package Usage:
- time: Rate limiting delays and timing measurements
- json: Serialize/deserialize search results to/from JSON format
- logging: Log API calls, errors, and debug information
- os: Access environment variables (e.g., API keys)
- pathlib.Path: Handle file paths for .env file loading
- typing: Type hints for function parameters and return values
- urllib.parse.quote: URL-encode search queries for API requests
- concurrent.futures: Parallel execution of database searches using ThreadPoolExecutor
- threading.Lock: Thread-safe rate limiting to prevent race conditions
- requests: HTTP requests to DBLP and Semantic Scholar REST APIs
- arxiv: Official arXiv API client for searching arXiv papers
- fuzzywuzzy.fuzz: Calculate title similarity scores using fuzzy string matching
- dotenv (optional): Load API keys from .env file instead of hardcoding them
"""

import time  # Rate limiting delays and timing measurements
import json  # Serialize/deserialize search results to/from JSON format
import logging  # Log API calls, errors, and debug information
import os  # Access environment variables (e.g., API keys)
from pathlib import Path  # Handle file paths for .env file loading
from typing import Dict, List, Optional, Any, Union  # Type hints for function parameters and return values
from urllib.parse import quote  # URL-encode search queries for API requests
from concurrent.futures import ThreadPoolExecutor, as_completed  # Parallel execution of database searches using ThreadPoolExecutor
from threading import Lock  # Thread-safe rate limiting to prevent race conditions
import requests  # HTTP requests to DBLP and Semantic Scholar REST APIs
import arxiv  # Official arXiv API client for searching arXiv papers
from fuzzywuzzy import fuzz  # Calculate title similarity scores using fuzzy string matching

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv  # Load API keys from .env file instead of hardcoding them
    # Load environment variables from .env file if it exists
    load_dotenv()
except ImportError:
    # python-dotenv is optional, continue without it
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limit constants (in seconds)
DBLP_RATE_LIMIT = 3.0
ARXIV_RATE_LIMIT = 3.0
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0

# API endpoints
DBLP_API_BASE = "https://dblp.org/search/publ/api"
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
# For Arxiv just package needed.

# Default similarity threshold for title matching (0-100)
DEFAULT_SIMILARITY_THRESHOLD = 80


class RateLimiter:
    """
    Thread-safe rate limiter to ensure API rate limits are respected.
    Each API source has its own rate limiter instance.
    """
    
    def __init__(self, min_interval: float):
        """
        Initialize rate limiter.
        
        Args:
            min_interval: Minimum time interval between requests in seconds
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.lock = Lock()
    
    def wait_if_needed(self):
        """
        Block until enough time has passed since the last request.
        Thread-safe implementation.
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


# Global rate limiters for each API
dblp_rate_limiter = RateLimiter(DBLP_RATE_LIMIT)
arxiv_rate_limiter = RateLimiter(ARXIV_RATE_LIMIT)
semantic_scholar_rate_limiter = RateLimiter(SEMANTIC_SCHOLAR_RATE_LIMIT)


def get_semantic_scholar_api_key() -> Optional[str]:
    """
    Load Semantic Scholar API key from environment variable or .env file.
    
    The API key can be set in two ways:
    1. As an environment variable: SEMANTIC_SCHOLAR_API_KEY
    2. In a .env file in the project root: SEMANTIC_SCHOLAR_API_KEY=your_key_here
    
    Returns:
        API key string if found, None otherwise
    """
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    
    if not api_key:
        logger.warning(
            "Semantic Scholar API key not found. "
            "Set SEMANTIC_SCHOLAR_API_KEY environment variable or add it to .env file. "
            "Continuing without API key (rate limits may apply)."
        )
    else:
        logger.debug("Semantic Scholar API key loaded successfully")
    
    return api_key


# Load API key at module level (optional, will be None if not set)
SEMANTIC_SCHOLAR_API_KEY = get_semantic_scholar_api_key()


def normalize_title(title: str) -> str:
    """
    Normalize title for comparison by converting to lowercase and removing extra whitespace.
    
    Args:
        title: Original title string
        
    Returns:
        Normalized title string
    """
    if not title:
        return ""
    return " ".join(title.lower().split())


def calculate_title_similarity(title1: str, title2: str) -> int:
    """
    Calculate similarity score between two titles using fuzzy matching.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Similarity score (0-100)
    """
    normalized1 = normalize_title(title1)
    normalized2 = normalize_title(title2)
    return fuzz.ratio(normalized1, normalized2)


def search_dblp(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search DBLP database for papers matching the given title.
    Only returns Open-Access papers (papers with accessible URLs).
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    dblp_rate_limiter.wait_if_needed()
    
    try:
        # URL encode the title for the query
        encoded_title = quote(title)
        url = f"{DBLP_API_BASE}?q={encoded_title}&format=json&h={max_results}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        papers = []
        hits = data.get('result', {}).get('hits', {})
        hit_list = hits.get('hit', [])
        
        # Handle both single hit (dict) and multiple hits (list) cases
        if isinstance(hit_list, dict):
            hit_list = [hit_list]
        elif not isinstance(hit_list, list):
            hit_list = []
        
        for hit in hit_list:
            info = hit.get('info', {})
            paper_title = info.get('title', '')
            
            # DBLP doesn't have explicit Open-Access flag, but we check for accessible URLs
            # Papers with URLs are considered accessible
            url = info.get('url', '')
            ee_links = info.get('ee', [])
            
            # Consider paper accessible if it has a URL or EE link
            is_accessible = bool(url) or bool(ee_links)
            
            if is_accessible:
                # Extract authors
                authors = []
                authors_data = info.get('authors', {})
                if isinstance(authors_data, dict):
                    author_list = authors_data.get('author', [])
                    if isinstance(author_list, dict):
                        author_list = [author_list]
                    for author in author_list:
                        if isinstance(author, dict):
                            authors.append(author.get('text', ''))
                        else:
                            authors.append(str(author))
                
                # Extract DOI from EE links
                doi = ""
                if isinstance(ee_links, list):
                    for link in ee_links:
                        if isinstance(link, dict) and 'doi.org' in link.get('text', ''):
                            doi = link.get('text', '')
                            break
                elif isinstance(ee_links, dict) and 'doi.org' in ee_links.get('text', ''):
                    doi = ee_links.get('text', '')
                
                # Get primary URL
                paper_url = url
                if not paper_url and ee_links:
                    if isinstance(ee_links, list) and ee_links:
                        paper_url = ee_links[0].get('text', '') if isinstance(ee_links[0], dict) else str(ee_links[0])
                    elif isinstance(ee_links, dict):
                        paper_url = ee_links.get('text', '')
                
                papers.append({
                    'title': paper_title,
                    'authors': authors,
                    'year': info.get('year', ''),
                    'doi': doi,
                    'url': paper_url,
                    'source': 'dblp',
                    'venue': info.get('venue', '')
                })
        
        logger.debug(f"DBLP search for '{title[:60]}...' returned {len(papers)} accessible papers")
        return papers
        
    except requests.exceptions.RequestException as e:
        logger.error(f"DBLP API error for title '{title[:60]}...': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying DBLP for '{title[:60]}...': {e}")
        return []


def search_arxiv(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search arXiv database for papers matching the given title.
    All arXiv papers are Open-Access by default.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    arxiv_rate_limiter.wait_if_needed()
    
    try:
        client = arxiv.Client(
            page_size=min(max_results, 10),
            delay_seconds=0,  # We handle rate limiting ourselves
            num_retries=3
        )
        
        search = arxiv.Search(
            query=title,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            # All arXiv papers are Open-Access
            authors = [author.name for author in result.authors]
            
            # Extract DOI if available
            doi = ""
            if result.doi:
                doi = result.doi
            
            papers.append({
                'title': result.title,
                'authors': authors,
                'year': result.published.year if result.published else None,
                'doi': doi,
                'url': result.entry_id,  # arXiv entry URL
                'pdf_url': result.pdf_url,
                'source': 'arxiv',
                'arxiv_id': result.get_short_id()
            })
        
        logger.debug(f"arXiv search for '{title[:60]}...' returned {len(papers)} papers")
        return papers
        
    except Exception as e:
        logger.error(f"Error querying arXiv for '{title[:60]}...': {e}")
        return []


def search_semantic_scholar(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar database for papers matching the given title.
    Only returns Open-Access papers (papers with openAccessPdf field).
    
    Uses API key if available (set via SEMANTIC_SCHOLAR_API_KEY environment variable
    or .env file) for higher rate limits.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    semantic_scholar_rate_limiter.wait_if_needed()
    
    try:
        params = {
            "query": title,
            "limit": max_results,
            "fields": "title,authors,year,externalIds,url,openAccessPdf,citationCount"
        }
        
        # Add API key to headers if available
        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
            logger.debug("Using Semantic Scholar API key for authentication")
        else:
            logger.debug("No API key provided, using public rate limits")
        
        response = requests.get(
            SEMANTIC_SCHOLAR_API_BASE,
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        papers = []
        paper_list = data.get('data', [])
        
        for paper in paper_list:
            # Only include papers with Open-Access PDF
            open_access_pdf = paper.get('openAccessPdf')
            if not open_access_pdf:
                continue
            
            # Extract authors
            authors = []
            for author in paper.get('authors', []):
                author_name = author.get('name', '')
                if author_name:
                    authors.append(author_name)
            
            # Extract DOI
            doi = ""
            external_ids = paper.get('externalIds', {})
            if external_ids:
                doi = external_ids.get('DOI', '')
            
            papers.append({
                'title': paper.get('title', ''),
                'authors': authors,
                'year': paper.get('year'),
                'doi': doi,
                'url': paper.get('url', ''),
                'pdf_url': open_access_pdf.get('url', '') if isinstance(open_access_pdf, dict) else '',
                'source': 'semantic_scholar',
                'paper_id': paper.get('paperId', ''),
                'citation_count': paper.get('citationCount', 0)
            })
        
        logger.debug(f"Semantic Scholar search for '{title[:60]}...' returned {len(papers)} Open-Access papers")
        return papers
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Semantic Scholar API error for title '{title[:60]}...': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying Semantic Scholar for '{title[:60]}...': {e}")
        return []


def filter_and_rank_results(
    original_title: str,
    all_results: List[Dict[str, Any]],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Filter results by title similarity and rank them.
    
    Args:
        original_title: Original title to compare against
        all_results: List of all paper results from all sources
        similarity_threshold: Minimum similarity score (0-100) to include a result
        
    Returns:
        List of filtered and ranked papers with similarity scores
    """
    filtered_results = []
    
    for paper in all_results:
        paper_title = paper.get('title', '')
        if not paper_title:
            continue
        
        similarity = calculate_title_similarity(original_title, paper_title)
        
        if similarity >= similarity_threshold:
            paper['similarity_score'] = similarity
            filtered_results.append(paper)
    
    # Sort by similarity score (descending)
    filtered_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return filtered_results


def search_papers_by_title(
    title: str,
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Search for papers across DBLP, arXiv, and Semantic Scholar databases.
    Results are filtered by Open-Access availability and title similarity.
    
    Args:
        title: Paper title to search for
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source
        parallel: Whether to search databases in parallel (recommended for efficiency)
        
    Returns:
        Dictionary containing:
            - 'original_title': The searched title
            - 'results': List of matching papers with metadata
            - 'summary': Summary statistics
    """
    logger.info(f"Searching for papers with title: '{title[:80]}...'")
    
    if parallel:
        # Search all databases in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(search_dblp, title, max_results_per_source): 'dblp',
                executor.submit(search_arxiv, title, max_results_per_source): 'arxiv',
                executor.submit(search_semantic_scholar, title, max_results_per_source): 'semantic_scholar'
            }
            
            all_results = []
            for future in as_completed(futures):
                source = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Found {len(results)} papers from {source}")
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
    else:
        # Sequential search (slower but simpler)
        all_results = []
        
        dblp_results = search_dblp(title, max_results_per_source)
        all_results.extend(dblp_results)
        logger.info(f"Found {len(dblp_results)} papers from DBLP")
        
        arxiv_results = search_arxiv(title, max_results_per_source)
        all_results.extend(arxiv_results)
        logger.info(f"Found {len(arxiv_results)} papers from arXiv")
        
        semantic_scholar_results = search_semantic_scholar(title, max_results_per_source)
        all_results.extend(semantic_scholar_results)
        logger.info(f"Found {len(semantic_scholar_results)} papers from Semantic Scholar")
    
    # Filter and rank results by similarity
    filtered_results = filter_and_rank_results(title, all_results, similarity_threshold)
    
    # Create summary statistics
    summary = {
        'total_found': len(all_results),
        'total_matching': len(filtered_results),
        'by_source': {}
    }
    
    for result in filtered_results:
        source = result.get('source', 'unknown')
        summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
    
    logger.info(f"Found {len(filtered_results)} papers matching title (similarity >= {similarity_threshold}%)")
    
    return {
        'original_title': title,
        'results': filtered_results,
        'summary': summary
    }


def search_multiple_titles(
    titles: List[str],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for multiple titles in parallel.
    Each title is searched across all databases, and titles are processed in parallel.
    
    Args:
        titles: List of paper titles to search for
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source per title
        max_workers: Maximum number of titles to process in parallel
        
    Returns:
        List of result dictionaries, one per title
    """
    logger.info(f"Searching {len(titles)} titles in parallel (max_workers={max_workers})")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                search_papers_by_title,
                title,
                similarity_threshold,
                max_results_per_source,
                parallel=True
            ): title
            for title in titles
        }
        
        for future in as_completed(futures):
            title = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                logger.info(f"Completed search for: '{title[:60]}...'")
            except Exception as e:
                logger.error(f"Error searching for '{title[:60]}...': {e}")
                all_results.append({
                    'original_title': title,
                    'results': [],
                    'summary': {'error': str(e)}
                })
    
    return all_results


def save_results_to_json(results: Union[Dict[str, Any], List[Dict[str, Any]]], output_file: str):
    """
    Save search results to a JSON file.
    
    Args:
        results: Single result dictionary or list of result dictionaries
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")


def main():
    """
    Example usage of the API caller.
    """
    # Example: Search for a single paper
    print("=" * 80)
    print("Example: Searching for a single paper")
    print("=" * 80)
    
    example_title = "Attention Is All You Need"
    result = search_papers_by_title(example_title, similarity_threshold=80)
    
    print(f"\nOriginal title: {result['original_title']}")
    print(f"Found {result['summary']['total_matching']} matching papers")
    print(f"\nTop matches:")
    for i, paper in enumerate(result['results'][:3], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Source: {paper['source']}")
        print(f"   Similarity: {paper['similarity_score']}%")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   DOI: {paper.get('doi', 'N/A')}")
        print(f"   URL: {paper.get('url', 'N/A')}")
    
    # Save results
    save_results_to_json(result, 'api_search_results.json')
    
    # Example: Search for multiple papers
    print("\n" + "=" * 80)
    print("Example: Searching for multiple papers")
    print("=" * 80)
    
    example_titles = [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners"
    ]
    
    multiple_results = search_multiple_titles(example_titles, max_workers=2)
    print(f"\nSearched {len(example_titles)} titles")
    print(f"Found results for {sum(1 for r in multiple_results if r['results'])} titles")
    
    save_results_to_json(multiple_results, 'api_search_results_multiple.json')


if __name__ == '__main__':
    main()

