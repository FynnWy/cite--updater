"""
Semantic Scholar API integration for citation verification task.

This script demonstrates how to:
1. Query Semantic Scholar API for authors by name
2. Query Semantic Scholar API for papers by title
3. Compare authors with verified results

Students should extend this to handle all citations and integrate with the main verification workflow.
"""

import json
import time
import os
from typing import Dict, List, Optional
import requests
from fuzzywuzzy import fuzz


def query_semantic_scholar_author(author_name: str, limit: int = 10) -> Optional[Dict]:
    """
    Query Semantic Scholar API for authors by name.

    Args:
        author_name: Author name to search for
        limit: Maximum number of results to return

    Returns:
        Dictionary with author info if found, None otherwise.
    """
    base_url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": author_name,
        "limit": limit,
        "fields": "name,authorId,paperCount,citationCount,hIndex,affiliations,homepage"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        authors = data.get('data', [])
        if authors:
            # Return the best match (first result)
            best_match = authors[0]
            return {
                "name": best_match.get("name"),
                "author_id": best_match.get("authorId"),
                "paper_count": best_match.get("paperCount"),
                "citation_count": best_match.get("citationCount"),
                "h_index": best_match.get("hIndex"),
                "affiliations": best_match.get("affiliations", []),
                "homepage": best_match.get("homepage"),
                "all_matches": authors,
                "source": "semantic_scholar"
            }
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Semantic Scholar API error for author '{author_name}': {e}")
        return None
    except Exception as e:
        print(f"⚠️  Unexpected error querying Semantic Scholar for author '{author_name}': {e}")
        return None


def query_semantic_scholar_paper(title: str, max_results: int = 5) -> Optional[Dict]:
    """
    Query Semantic Scholar API for a paper by title similarity.

    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return

    Returns:
        Dictionary with paper info if found, or None if not.
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": max_results,
        "fields": "title,authors,year,venue,abstract,citationCount,referenceCount,paperId"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        papers = data.get('data', [])
        if not papers:
            return None
        
        def normalize(s: str) -> str:
            return s.strip().lower().replace("-", " ").replace("_", " ")
        
        def similarity(a: str, b: str) -> float:
            return fuzz.ratio(normalize(a), normalize(b)) / 100.0
        
        # Find best matching paper by title similarity
        best_match = None
        best_score = 0
        
        for paper in papers:
            paper_title = paper.get("title", "")
            if paper_title:
                score = similarity(title, paper_title)
                if score > best_score:
                    best_score = score
                    best_match = paper
        
        # Good match threshold
        if best_match and best_score >= 0.80:
            # Extract author names from the paper data
            authors = []
            for author in best_match.get("authors", []):
                author_name = author.get("name", "")
                if author_name:
                    authors.append(author_name)
            
            return {
                "title": best_match.get("title"),
                "authors": authors,
                "year": best_match.get("year"),
                "venue": best_match.get("venue"),
                "paper_id": best_match.get("paperId"),
                "citation_count": best_match.get("citationCount"),
                "reference_count": best_match.get("referenceCount"),
                "match_score": round(best_score * 100, 1),
                "source": "semantic_scholar"
            }
        
        # No good match found
        return {
            "title": title,
            "authors": [],
            "year": None,
            "venue": None,
            "paper_id": None,
            "citation_count": None,
            "reference_count": None,
            "match_score": round(best_score * 100, 1),
            "best_match_title": best_match.get("title") if best_match else None,
            "best_match_score": round(best_score * 100, 1),
            "source": "semantic_scholar_no_match"
        }
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Semantic Scholar API error for title '{title[:60]}...': {e}")
        return None
    except Exception as e:
        print(f"⚠️  Unexpected error querying Semantic Scholar for title '{title[:60]}...': {e}")
        return None


def compare_authors(original_authors: List[str], verified_authors: List[str]) -> Dict:
    """
    Compare original authors with verified authors based on first, middle, and last name similarity.
    - Order is ignored.
    - Minor formatting differences (punctuation, initials) tolerated.
    - Abbreviations like "M." or "M" for a full first name are acceptable.
    - Missing or extra middle names are treated as discrepancies.
    This function is defensive: it always returns a dict with 'match' and 'discrepancies'.
    """
    # Defensive defaults
    discrepancies = []
    matched_original = set()
    matched_verified = set()

    try:
        def normalize(name: str) -> str:
            if not isinstance(name, str):
                return ""
            return name.strip().lower().replace('.', '').replace(',', '')

        def split_name(full_name: str):
            """
            Splits a full name into (given_name_parts_list, last_name)
            Example:
                "M Pawan Kumar" -> (["m","pawan"], "kumar")
                "M. Kumar" -> (["m"], "kumar")
            """
            s = normalize(full_name)
            parts = s.split()
            if len(parts) == 0:
                return [], ""
            if len(parts) == 1:
                return [parts[0]], ""
            return parts[:-1], parts[-1]

        def name_similarity(a: str, b: str) -> float:
            # handle non-string gracefully
            a = a or ""
            b = b or ""
            return fuzz.ratio(a, b) / 100.0

        def is_initial(token: str) -> bool:
            return isinstance(token, str) and len(token) == 1

        def is_firstname_match(given_a: list, given_b: list) -> bool:
            """
            Check if the given-name parts (first + middle) match according to these rules:
            - First names must match closely (exact, via initial, or >=0.85 similarity)
            - Initials for the first name are acceptable
            - Missing or extra middle names count as discrepancy (per user requirement)
            """
            # ensure list types
            if not isinstance(given_a, list) or not isinstance(given_b, list):
                return False
            if not given_a or not given_b:
                return False

            first_a = given_a[0]
            first_b = given_b[0]

            # First name check
            def first_match_ok(a, b):
                if not a or not b:
                    return False
                if a == b:
                    return True
                if is_initial(a) and b.startswith(a):
                    return True
                if is_initial(b) and a.startswith(b):
                    return True
                return name_similarity(a, b) >= 0.85

            if not first_match_ok(first_a, first_b):
                return False

            # Middle name comparison
            middle_a = given_a[1:]
            middle_b = given_b[1:]

            # If one has middle names and the other doesn't -> discrepancy
            if (len(middle_a) > 0 and len(middle_b) == 0) or (len(middle_b) > 0 and len(middle_a) == 0):
                return False

            # If both have middle names -> lengths must match and each part pairwise similar
            if len(middle_a) > 0 and len(middle_b) > 0:
                if len(middle_a) != len(middle_b):
                    return False
                for ma, mb in zip(middle_a, middle_b):
                    if ma == mb:
                        continue
                    if is_initial(ma) and mb.startswith(ma):
                        continue
                    if is_initial(mb) and ma.startswith(mb):
                        continue
                    if name_similarity(ma, mb) >= 0.85:
                        continue
                    return False

            return True

        def is_name_match(original: str, verified: str) -> bool:
            """
            Combine last name check and given-name check.
            Last name must be very similar (>=0.9),
            given name (including middle names) checked via is_firstname_match().
            """
            o_given, o_last = split_name(original)
            v_given, v_last = split_name(verified)

            # Last name similarity required
            if not o_last and not v_last:
                last_ok = True
            else:
                last_ok = name_similarity(o_last, v_last) >= 0.9

            if not last_ok:
                return False

            return is_firstname_match(o_given, v_given)

        # --- Perform comparison ---
        for v_auth in verified_authors or []:
            try:
                # find exact matches according to is_name_match
                matches = [o_auth for o_auth in (original_authors or []) if is_name_match(o_auth, v_auth)]
            except Exception:
                matches = []

            if matches:
                matched_verified.add(v_auth)
                # mark first matched original as used
                matched_original.add(matches[0])
            else:
                # find best original by simple string similarity for reporting
                best_original = None
                best_score = 0.0
                for o_auth in (original_authors or []):
                    score = name_similarity(normalize(o_auth), normalize(v_auth))
                    if score > best_score:
                        best_score = score
                        best_original = o_auth

                if best_original:
                    discrepancies.append({
                        "type": "missing_author",
                        "details": f"Missing or mismatched author: {v_auth} - {best_original}",
                        "best_original": best_original,
                        "similarity": round(best_score, 3)
                    })
                else:
                    discrepancies.append({
                        "type": "missing_author",
                        "details": f"Missing or mismatched author: {v_auth}",
                        "best_original": None,
                        "similarity": 0.0
                    })

        return {
            "match": len(discrepancies) == 0,
            "discrepancies": discrepancies
        }

    except Exception as e:
        # Fallback: ensure function always returns a dict and include the exception message for debugging
        return {
            "match": False,
            "discrepancies": [
                {
                    "type": "internal_error",
                    "details": f"compare_authors crashed: {repr(e)}"
                }
            ]
        }


def verify_citation_semantic_scholar(citation: Dict) -> Dict:
    """
    Verify a single citation using Semantic Scholar API.
    Prints clean, readable verification status.
    """
    print(f"\nVerifying: {citation['title'][:60]}...")

    scholar_result = query_semantic_scholar_paper(citation['title'])

    # No results → Simple error message
    if not scholar_result or scholar_result.get("source") == "semantic_scholar_no_match":
        best = scholar_result.get("best_match_title") if scholar_result else "N/A"
        score = scholar_result.get("best_match_score") if scholar_result else 0
        print(f"❌ No good Semantic Scholar match found for: '{citation['title']}' (best match: '{best}', {score}%)")
        return {
            "status": "not_found",
            "message": f"No good Semantic Scholar match found ({score}%)",
            "best_title": best,
            "match_score": score
        }

    # Result → Author comparison
    comparison = compare_authors(
        citation.get("authors", []),
        scholar_result["authors"]
    )

    result = {
        "status": "verified" if comparison["match"] else "discrepancy_found",
        "source": "semantic_scholar",
        "verified_authors": scholar_result["authors"],
        "match_score": scholar_result["match_score"],
        "comparison": comparison
    }

    # If match was found
    if comparison["match"]:
        print(f"✅ Title found in Semantic Scholar for: \"{scholar_result['title']}\"")
        print("  ✓ Authors match!")
    else:
        print(f"✅ Title found in Semantic Scholar for: \"{scholar_result['title']}\"")
        print("  ⚠️  Discrepancies found:")
        for disc in comparison["discrepancies"]:
            print(f"     - {disc['details']}")

    return result


def main():
    """Main function - example usage."""
    # Example: Search for an author
    print("="*60)
    print("Example: Searching for an author in Semantic Scholar")
    print("="*60)
    author_result = query_semantic_scholar_author("Yann LeCun")
    if author_result:
        print(f"\n✅ Found author: {author_result['name']}")
        print(f"   Author ID: {author_result['author_id']}")
        print(f"   Papers: {author_result['paper_count']}")
        print(f"   Citations: {author_result['citation_count']}")
    else:
        print("\n❌ Author not found")
    
    # Example: Load and verify citations
    print("\n" + "="*60)
    print("Example: Verifying citations using Semantic Scholar")
    print("="*60)
    
    citations_file = 'citations.json'
    if not os.path.exists(citations_file):
        print(f"⚠️  Citations file not found: {citations_file}")
        print("   Skipping citation verification example")
        return
    
    print("Loading citations...")
    with open(citations_file, 'r', encoding='utf-8') as f:
        citations = json.load(f)
    print(f"Loaded {len(citations)} citations")
    
    # Verify first n citations as an example
    number = 5
    print("\n" + "="*60)
    print(f"Verifying first {number} citations...")
    print("="*60)
    
    results = []
    for i, citation in enumerate(citations[:number], 1):
        print(f"\n[{i}/{number}]")
        result = verify_citation_semantic_scholar(citation)
        results.append({
            'original': citation,
            'verification': result
        })
        
        # Be respectful - add delay between API calls (Semantic Scholar allows 100 requests per 5 minutes)
        time.sleep(1)
    
    # Save results
    output_file = 'semantic_scholar_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"Example complete! Results saved to '{output_file}'")
    print("="*60)
    print("\nNote: Semantic Scholar API rate limit is 100 requests per 5 minutes")
    print("      Consider adding delays or using an API key for higher limits")


if __name__ == '__main__':
    main()

