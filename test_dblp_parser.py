from src.parser.dblp_parser import DblpParser
import time
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_search(parser: DblpParser, titles: List[Dict[str, str]], threshold: float = 5.0) -> None:
    """
    Test search functionality with multiple titles and report results
    
    Args:
        parser: Initialized DblpParser instance
        titles: List of test cases with titles and optional metadata
        threshold: BM25 score threshold for matches
    """
    total_time = 0
    found_count = 0
    
    for test_case in titles:
        title = test_case["title"]
        logger.info(f"\n{'='*80}")
        logger.info(f"Searching for: {title}")
        
        start_time = time.time()
        result = parser.search_by_title(title, threshold=threshold)
        search_time = time.time() - start_time
        total_time += search_time
        
        if result:
            found_count += 1
            logger.info(f"✓ Found publication in {search_time:.3f} seconds:")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Authors: {', '.join(result['authors'])}")
            logger.info(f"Year: {result['year']}")
            logger.info(f"Venue: {result['venue']}")
            if result['doi']:
                logger.info(f"DOI: {result['doi']}")
        else:
            logger.info(f"✗ Publication not found ({search_time:.3f} seconds)")
    
    # Print summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("Search Summary:")
    logger.info(f"Total searches: {len(titles)}")
    logger.info(f"Found: {found_count}")
    logger.info(f"Not found: {len(titles) - found_count}")
    logger.info(f"Average search time: {total_time/len(titles):.3f} seconds")
    logger.info(f"Total time: {total_time:.3f} seconds")

def main():
    xml_path = "dblp/dblp-2024-11-04.xml"
    
    # Initialize the parser with cache directory
    logger.info("Initializing DBLP parser...")
    parser = DblpParser(xml_path, cache_dir="dblp_cache")
    
    # Test cases - famous papers that should be in DBLP
    test_cases = [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "expected_year": "2019",
            "expected_venue": "NAACL"
        },
        {
            "title": "Attention Is All You Need",
            "expected_year": "2017",
            "expected_venue": "NIPS"
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "expected_year": "2016",
            "expected_venue": "CVPR"
        },
        # Add more test cases for different scenarios
        {
            "title": "GPT: Improving Language Understanding by Generative Pre-Training",
            "expected_year": "2018"
        },
        {
            "title": "Adam: A Method for Stochastic Optimization",
            "expected_year": "2015",
            "expected_venue": "ICLR"
        }
    ]
    
    # Run tests with different thresholds
    thresholds = [3.0, 5.0, 7.0]
    for threshold in thresholds:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Testing with threshold: {threshold}")
        test_search(parser, test_cases, threshold=threshold)

if __name__ == "__main__":
    main() 