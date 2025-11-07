"""
Example starter script for citation verification task.

This script demonstrates how to:
1. Load citations from JSON
2. Query arXiv API for a single citation
3. Compare authors

Students should extend this to handle all citations and use multiple APIs.
"""

import json
import arxiv
import time
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz


def load_citations(json_file: str) -> List[Dict]:
    """Load citations from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def query_arxiv_by_title(title: str, max_results: int = 5) -> Optional[Dict]:
    """
    Query arXiv API for a paper by title.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with paper info if found, None otherwise
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=title,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        best_match = None
        best_score = 0
        
        for result in client.results(search):
            # Calculate title similarity
            score = fuzz.ratio(result.title.lower(), title.lower())
            print(f"  arXiv result: '{result.title[:60]}...' (similarity: {score}%)")
            
            if score > best_score:
                best_score = score
                best_match = result
        
        if best_match and best_score >= 80:  # Threshold for good match
            return {
                'title': best_match.title,
                'authors': [author.name for author in best_match.authors],
                'year': best_match.published.year if best_match.published else None,
                'match_score': best_score
            }
        
        return None
        
    except Exception as e:
        print(f"  Error querying arXiv: {e}")
        return None


def compare_authors(original_authors: List[str], verified_authors: List[str]) -> Dict:
    """
    Compare original authors with verified authors.
    
    Returns a dictionary with comparison results.
    """
    discrepancies = []
    
    # Check for missing authors
    missing = [a for a in verified_authors if a not in original_authors]
    if missing:
        discrepancies.append({
            'type': 'missing_author',
            'details': f"Missing authors: {', '.join(missing)}"
        })
    
    # Check for extra authors (might be incorrect in original)
    extra = [a for a in original_authors if a not in verified_authors]
    if extra:
        discrepancies.append({
            'type': 'extra_author',
            'details': f"Extra authors in original: {', '.join(extra)}"
        })
    
    # Check order (simplified - just check if first author matches)
    if original_authors and verified_authors:
        if original_authors[0] != verified_authors[0]:
            discrepancies.append({
                'type': 'wrong_order',
                'details': f"First author mismatch: '{original_authors[0]}' vs '{verified_authors[0]}'"
            })
    
    return {
        'match': len(discrepancies) == 0,
        'discrepancies': discrepancies
    }


def verify_citation(citation: Dict) -> Dict:
    """
    Verify a single citation by querying arXiv.
    
    Args:
        citation: Dictionary with citation information
        
    Returns:
        Dictionary with verification results
    """
    print(f"\nVerifying: {citation['title'][:60]}...")
    
    # Query arXiv
    arxiv_result = query_arxiv_by_title(citation['title'])
    
    if not arxiv_result:
        return {
            'status': 'not_found',
            'message': 'Paper not found in arXiv'
        }
    
    # Compare authors
    comparison = compare_authors(
        citation.get('authors', []),
        arxiv_result['authors']
    )
    
    result = {
        'status': 'verified' if comparison['match'] else 'discrepancy_found',
        'source': 'arxiv',
        'verified_authors': arxiv_result['authors'],
        'match_score': arxiv_result['match_score'],
        'comparison': comparison
    }
    
    if not comparison['match']:
        print(f"  ⚠️  Discrepancies found:")
        for disc in comparison['discrepancies']:
            print(f"     - {disc['details']}")
    else:
        print(f"  ✓ Authors match!")
    
    return result


def main():
    """Main function - example usage."""
    # Load citations
    print("Loading citations...")
    citations = load_citations('citations.json')
    print(f"Loaded {len(citations)} citations")
    
    # Verify first 5 citations as an example
    print("\n" + "="*60)
    print("Verifying first 5 citations (example)...")
    print("="*60)
    
    results = []
    for i, citation in enumerate(citations[:5], 1):
        print(f"\n[{i}/5]")
        result = verify_citation(citation)
        results.append({
            'original': citation,
            'verification': result
        })
        
        # Be respectful - add delay between API calls
        time.sleep(2)
    
    # Save results
    with open('example_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Example complete! Results saved to 'example_results.json'")
    print("="*60)
    print("\nNext steps:")
    print("1. Extend this to process all citations")
    print("2. Add DBLP and Semantic Scholar API queries")
    print("3. Implement better author name matching")
    print("4. Add progress saving and resume functionality")


if __name__ == '__main__':
    main()

