"""
Citation Author Validation Script

This script validates citation authors in parsed JSON files against selected
reference databases (DBLP, ACL Anthology, arXiv). For each reference, it
queries the configured sources by title and compares author names to detect
incorrect citations.

Usage:
    python -m src.name_matching.validate_citations [options]
    
Options:
    --input-dir DIR       Directory containing parsed JSON files (default: data/parsed_jsons)
    --dblp-xml FILE       Path to DBLP XML file (required when DBLP source is used)
    --sources LIST        Comma-separated sources: dblp,acl,arxiv (default: dblp,acl,arxiv)
    --output FILE         Output JSON file path (default: citation_validation_results.json)
    --num-files N         Number of JSON files to process (default: 20)
    --threshold N         Retrieval threshold (DBLP BM25; default: 5.0)
    --similarity-method   String similarity metric: fuzz/fuzz.ratio or damerau
"""

import json
import os
import logging
import argparse
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import Counter, defaultdict
import random
import re
from tqdm import tqdm

Path("data").mkdir(parents=True, exist_ok=True)

# Setup logging BEFORE importing other modules
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/citation_validation.log'),
        logging.StreamHandler()
    ]
)

# Reduce verbosity of external libraries BEFORE importing them
logging.getLogger('retriv').setLevel(logging.ERROR)
logging.getLogger('dblp_parser').setLevel(logging.ERROR)
# Also try to suppress all logging from retriv submodules
logging.getLogger('retriv.SparseRetriever').setLevel(logging.ERROR)

from .analyze_matches import (
    is_name_match,
    check_author_lists,
    initial_matches,
    is_compound_initial
)
from ..parser.dblp_parser import DblpParser
from .sources import (
    AVAILABLE_SOURCES,
    ACLAnthologySource,
    ArXivSource,
    CitationSource,
    DBLPSource,
    SourceMatch,
)
from nameparser import HumanName
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein
from unidecode import unidecode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_SOURCES = ",".join(AVAILABLE_SOURCES)
SIMILARITY_METHODS = ("fuzz", "damerau")
DEFAULT_SIMILARITY_METHOD = "fuzz"
DEFAULT_NAME_PART_FUZZ_THRESHOLD = 85.0
DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE = 1


def normalize_last_name_with_prefixes(last_name: str) -> tuple[str, str]:
    """
    Normalize last names by removing common prefixes (De, Van, Von, etc.).
    Returns both the normalized version (without prefix) and the original.
    
    Args:
        last_name: Last name string
        
    Returns:
        Tuple of (normalized_last_name, original_last_name)
    """
    if not last_name:
        return '', ''
    
    # Common last name prefixes
    prefixes = ['de', 'van', 'von', 'del', 'della', 'di', 'da', 'le', 'la', 'el', 'der', 'den', 'du', 'des']
    
    last_lower = last_name.lower().strip()
    parts = last_lower.split()
    
    # Check if first part is a prefix
    if parts and parts[0] in prefixes:
        # Return name without prefix, but keep original for reference
        normalized = ' '.join(parts[1:]) if len(parts) > 1 else parts[0]
        return normalized, last_name
    
    return last_lower, last_name


def normalize_name_for_comparison(name: str) -> str:
    """
    Normalize a name component for comparison by:
    1. Converting to lowercase
    2. Removing accents using unidecode
    3. Removing periods and extra spaces
    
    Args:
        name: Name component string
        
    Returns:
        Normalized string for comparison
    """
    if not name:
        return ''
    # Convert to lowercase, remove accents, remove periods, strip whitespace
    normalized = unidecode(name.lower().replace('.', '').strip())
    return normalized


def names_match_with_accents(name1: str, name2: str) -> bool:
    """
    Check if two names match, considering accents/diacritics.
    Uses unidecode to normalize accents.
    
    Args:
        name1: First name string
        name2: Second name string
        
    Returns:
        True if names match (with or without accents)
    """
    if not name1 or not name2:
        return False
    
    # Direct match
    if name1.lower().strip() == name2.lower().strip():
        return True
    
    # Match after removing accents
    name1_normalized = normalize_name_for_comparison(name1)
    name2_normalized = normalize_name_for_comparison(name2)
    
    return name1_normalized == name2_normalized


def handle_middle_initial_match(ref_author: Dict[str, str], dblp_author: Dict[str, str]) -> bool:
    """
    Check if authors match when one has a middle initial and the other doesn't.
    For example: "Ed Chi" vs "Ed H. Chi" should match.
    
    Args:
        ref_author: Reference author dictionary
        dblp_author: DBLP author dictionary
        
    Returns:
        True if authors match considering middle initials
    """
    ref_first = normalize_name_for_comparison(ref_author.get('first_name', ''))
    ref_middle = normalize_name_for_comparison(ref_author.get('middle_name', ''))
    ref_last = normalize_name_for_comparison(ref_author.get('last_name', ''))
    
    dblp_first = normalize_name_for_comparison(dblp_author.get('first_name', ''))
    dblp_middle = normalize_name_for_comparison(dblp_author.get('middle_name', ''))
    dblp_last = normalize_name_for_comparison(dblp_author.get('last_name', ''))
    
    # Last names must match (with accent normalization)
    if not names_match_with_accents(ref_author.get('last_name', ''), dblp_author.get('last_name', '')):
        return False
    
    # Check if first names match
    if ref_first == dblp_first:
        # If first names match exactly, they match (middle initial doesn't matter)
        return True
    
    # Check if one first name is just an initial of the other
    ref_is_initial = len(ref_first.replace(' ', '')) == 1
    dblp_is_initial = len(dblp_first.replace(' ', '')) == 1
    
    if ref_is_initial and dblp_first:
        # ref is initial, dblp is full name - check if initial matches
        return ref_first[0] == dblp_first[0]
    elif dblp_is_initial and ref_first:
        # dblp is initial, ref is full name - check if initial matches
        return dblp_first[0] == ref_first[0]
    
    # Check if first name + middle name combination matches
    ref_full_first = f"{ref_first} {ref_middle}".strip()
    dblp_full_first = f"{dblp_first} {dblp_middle}".strip()
    
    if ref_full_first == dblp_full_first:
        return True
    
    # Check if one is a prefix of the other (e.g., "Ed" vs "Ed H")
    if ref_first and dblp_full_first.startswith(ref_first):
        return True
    if dblp_first and ref_full_first.startswith(dblp_first):
        return True
    
    return False


def calculate_title_similarity(
    title1: str,
    title2: str,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
) -> float:
    """
    Calculate title similarity score between 0 and 100.

    ``fuzz`` uses ``fuzz.ratio``.
    ``damerau`` uses normalized Damerau-Levenshtein distance converted to 0-100.
    """
    t1 = ' '.join(title1.lower().split())
    t2 = ' '.join(title2.lower().split())

    if similarity_method == "fuzz":
        return float(fuzz.ratio(t1, t2))

    if similarity_method == "damerau":
        max_len = max(len(t1), len(t2))
        if max_len == 0:
            return 100.0
        distance = DamerauLevenshtein.distance(t1, t2)
        return max(0.0, (1.0 - (distance / max_len)) * 100.0)

    raise ValueError(
        f"Unsupported similarity method: {similarity_method}. "
        f"Expected one of: {', '.join(SIMILARITY_METHODS)}"
    )


def normalize_name_parts(text: str) -> List[str]:
    """
    Normalize name text into alphabetic parts.

    This mirrors the historical part-based normalization used in
    ``analyze_matches.py`` before Damerau distance checks.
    """
    normalized = unidecode((text or "").lower())
    parts = normalized.replace('-', ' ').split()
    cleaned_parts = [''.join(c for c in part if c.isalpha()) for part in parts]
    return [part for part in cleaned_parts if part]


def parts_are_similar(
    part1: str,
    part2: str,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
    fuzz_threshold: float = DEFAULT_NAME_PART_FUZZ_THRESHOLD,
    damerau_max_distance: int = DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
) -> bool:
    """
    Compare two normalized name parts via fuzz ratio or Damerau-Levenshtein.

    The Damerau branch reuses the historical repository logic:
    - ignore one-character fuzziness
    - reject if length difference exceeds ``max_distance``
    - accept if ``DamerauLevenshtein.distance <= max_distance``
    """
    if not part1 or not part2:
        return False

    if len(part1) <= 1 or len(part2) <= 1:
        return part1 == part2

    if similarity_method == "fuzz":
        return fuzz.ratio(part1, part2) >= fuzz_threshold

    if similarity_method == "damerau":
        if abs(len(part1) - len(part2)) > damerau_max_distance:
            return False
        return DamerauLevenshtein.distance(part1, part2) <= damerau_max_distance

    raise ValueError(
        f"Unsupported similarity method: {similarity_method}. "
        f"Expected one of: {', '.join(SIMILARITY_METHODS)}"
    )


def names_match_by_parts(
    ref_author: Dict[str, str],
    dblp_author: Dict[str, str],
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
    fuzz_threshold: float = DEFAULT_NAME_PART_FUZZ_THRESHOLD,
    damerau_max_distance: int = DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
) -> bool:
    """
    Compare given-name parts (first+middle) with configurable similarity method.

    Last names are intentionally not checked here and should be checked separately.
    """
    ref_given = f"{ref_author.get('first_name', '')} {ref_author.get('middle_name', '')}".strip()
    dblp_given = f"{dblp_author.get('first_name', '')} {dblp_author.get('middle_name', '')}".strip()

    ref_parts = normalize_name_parts(ref_given)
    dblp_parts = normalize_name_parts(dblp_given)

    if not ref_parts or not dblp_parts:
        return False

    # Compare each token in the shorter side against the longer side.
    if len(ref_parts) > len(dblp_parts):
        ref_parts, dblp_parts = dblp_parts, ref_parts

    for ref_part in ref_parts:
        if not any(
            parts_are_similar(
                ref_part,
                dblp_part,
                similarity_method=similarity_method,
                fuzz_threshold=fuzz_threshold,
                damerau_max_distance=damerau_max_distance,
            )
            for dblp_part in dblp_parts
        ):
            return False
    return True


def check_author_with_minimum_lists(
    ref_authors: List[Dict[str, str]],
    dblp_authors: List[Dict[str, str]],
    title: str,
    max_authors: int = 10,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
    name_part_fuzz_threshold: float = DEFAULT_NAME_PART_FUZZ_THRESHOLD,
    name_part_damerau_max_distance: int = DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
) -> Dict[str, Any]:
    """
    Check author lists by comparing only the first N authors (default: 10).
    This handles cases where authors don't include the full author list.
    
    Args:
        ref_authors: Normalized authors from reference
        dblp_authors: Normalized authors from DBLP
        title: Paper title for context
        max_authors: Maximum number of authors to compare (default: 10)
        similarity_method: ``fuzz`` or ``damerau`` for part-based fallback matching
        name_part_fuzz_threshold: Minimum fuzz ratio for part matching when method=fuzz
        name_part_damerau_max_distance: Maximum Damerau distance for part matching when method=damerau
        
    Returns:
        Dictionary with:
            - matches: Boolean indicating if authors match
            - mismatches: List of mismatch descriptions
            - error_classifications: List of error types found
    """
    result = {
        'matches': False,
        'mismatches': [],
        'error_classifications': []
    }
    
    # Take first max_authors from both lists (or fewer if lists are shorter)
    ref_subset = ref_authors[:max_authors]
    dblp_subset = dblp_authors[:max_authors]

    if not ref_subset:
        result['mismatches'].append('Empty author list')
        result['error_classifications'].append('empty_list')
        return result

    # Check for parsing errors first (names with "*" or other invalid characters)
    parsing_errors_found = []
    for i, ref_author in enumerate(ref_subset):
        if ref_author.get('parsing_error', False):
            parsing_errors_found.append(f"Reference author {i+1}: {ref_author.get('original', 'Unknown')}")
        # Also check for names that are likely fragments (very short, single word, no last name)
        ref_first = ref_author.get('first_name', '').strip()
        ref_last = ref_author.get('last_name', '').strip()
        original = ref_author.get('original', '').strip()
        # Check if name is a fragment: single short word (1-3 chars) with no last name
        if not ref_last and len(ref_first) <= 3 and len(original) <= 3:
            parsing_errors_found.append(f"Reference author {i+1}: {original} (likely fragment)")
        # Check for single-word fragments: if only last name exists without first name, it's likely a parsing error
        # (unless it's a very common single-name pattern, but we'll flag it anyway)
        if not ref_first and ref_last and len(original.split()) == 1:
            parsing_errors_found.append(f"Reference author {i+1}: {original} (single-word fragment)")
    for j, dblp_author in enumerate(dblp_subset):
        if dblp_author.get('parsing_error', False):
            parsing_errors_found.append(f"DBLP author {j+1}: {dblp_author.get('original', 'Unknown')}")
    
    if parsing_errors_found:
        result['error_classifications'].append('parsing_error')
        result['mismatches'].append(
            'Parsing error detected: invalid characters or fragments in author names. ' + 
            '; '.join(parsing_errors_found)
        )
        return result

    # Try to match authors - allow for order differences and length differences
    matched_ref_indices = set()
    matched_dblp_indices = set()
    matches = []

    # First pass: exact matches using is_name_match (handles initials, reversed names, etc.)
    # Also check for matches with accent normalization and middle initials
    # Compare all reference authors against all DBLP authors (within max_authors limit)
    for i, ref_author in enumerate(ref_subset):
        for j, dblp_author in enumerate(dblp_subset):
            if j in matched_dblp_indices:
                continue
            # Try standard name matching first
            if is_name_match(ref_author, dblp_author):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break
            
            # Check for accent-normalized matches (e.g., "Kuebler" vs "Kübler")
            ref_first = ref_author.get('first_name', '').strip()
            ref_last = ref_author.get('last_name', '').strip()
            dblp_first = dblp_author.get('first_name', '').strip()
            dblp_last = dblp_author.get('last_name', '').strip()
            
            # Check if names match after accent normalization
            if (names_match_with_accents(ref_first, dblp_first) and 
                names_match_with_accents(ref_last, dblp_last)):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break
            
            # Check for compound last names with prefixes (De Choudhury vs Choudhury)
            ref_last_base, _ = normalize_last_name_with_prefixes(ref_last)
            dblp_last_base, _ = normalize_last_name_with_prefixes(dblp_last)
            if (names_match_with_accents(ref_first, dblp_first) and 
                (names_match_with_accents(ref_last_base, dblp_last_base) or
                 names_match_with_accents(ref_last, dblp_last_base) or
                 names_match_with_accents(ref_last_base, dblp_last))):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break
            
            # Also check for matches with middle initials (e.g., "Ed Chi" vs "Ed H. Chi")
            if handle_middle_initial_match(ref_author, dblp_author):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break

            # Historical fallback: part-based given-name comparison with configurable
            # fuzz/Damerau metric, only when last names already align.
            ref_last_base, _ = normalize_last_name_with_prefixes(ref_last)
            dblp_last_base, _ = normalize_last_name_with_prefixes(dblp_last)
            last_names_match = (
                names_match_with_accents(ref_last, dblp_last)
                or names_match_with_accents(ref_last_base, dblp_last_base)
                or names_match_with_accents(ref_last, dblp_last_base)
                or names_match_with_accents(ref_last_base, dblp_last)
            )
            if last_names_match and names_match_by_parts(
                ref_author,
                dblp_author,
                similarity_method=similarity_method,
                fuzz_threshold=name_part_fuzz_threshold,
                damerau_max_distance=name_part_damerau_max_distance,
            ):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break
    
    # Check for unmatched authors and classify errors
    unmatched_ref = [ref_subset[i] for i in range(len(ref_subset)) if i not in matched_ref_indices]
    unmatched_dblp = [dblp_subset[j] for j in range(len(dblp_subset)) if j not in matched_dblp_indices]
    
    if not unmatched_ref and not unmatched_dblp:
        result['matches'] = True
        return result
    
    # First, check for systematic parsing errors (names shifted/mixed up)
    # This happens when first names of one author become last names of another
    parsing_error_detected = False

    # Check for names that got split across multiple reference authors
    # Also check for names that are incorrectly combined (like "Yunpu Volker Tresp")
    for i, ref_author in enumerate(unmatched_ref):
        ref_first = ref_author.get('first_name', '').strip()
        ref_last = ref_author.get('last_name', '').strip()
        ref_original = ref_author.get('original', '').strip()
        
        # Check if a reference author has multiple words in first_name that look like separate names
        # Example: "Yunpu Volker" as first_name with "Tresp" as last_name
        if ref_first and ' ' in ref_first:
            first_parts = ref_first.split()
            # If first_name has 2+ words and last_name exists, this might be a parsing error
            # where two names were combined
            if len(first_parts) >= 2 and ref_last:
                # Check if this matches a DBLP author when split differently
                for j, dblp_author in enumerate(unmatched_dblp):
                    dblp_first = dblp_author.get('first_name', '').strip()
                    dblp_last = dblp_author.get('last_name', '').strip()
                    # If last names match and one of the first_name parts matches dblp_first
                    if ref_last.lower() == dblp_last.lower():
                        if any(part.lower() == dblp_first.lower() for part in first_parts):
                            parsing_error_detected = True
                            break
                if parsing_error_detected:
                    break
    
    # Check if DBLP author matches parts scattered across consecutive reference authors
    if not parsing_error_detected:
        for j, dblp_author in enumerate(unmatched_dblp):
            dblp_full = f"{dblp_author.get('first_name', '')} {dblp_author.get('middle_name', '')} {dblp_author.get('last_name', '')}".strip()
            dblp_parts = dblp_full.lower().split()

            # Check if this DBLP author matches parts scattered across consecutive reference authors
            for i in range(len(unmatched_ref) - 1):
                ref_author1 = unmatched_ref[i]
                ref_author2 = unmatched_ref[i + 1]

                ref1_full = f"{ref_author1.get('first_name', '')} {ref_author1.get('last_name', '')}".strip().lower()
                ref2_full = f"{ref_author2.get('first_name', '')} {ref_author2.get('last_name', '')}".strip().lower()

                combined = f"{ref1_full} {ref2_full}".replace('  ', ' ').strip()

                # Check if combined reference authors match the DBLP author
                if combined and dblp_full.lower() == combined:
                    parsing_error_detected = True
                    break

                # Check if parts of DBLP author are split between reference authors
                ref1_parts = ref1_full.split()
                ref2_parts = ref2_full.split()

                # If DBLP has 3+ parts and reference authors together have the same parts
                if len(dblp_parts) >= 3 and len(ref1_parts + ref2_parts) >= len(dblp_parts):
                    if all(part in ref1_parts + ref2_parts for part in dblp_parts):
                        parsing_error_detected = True
                        break

            if parsing_error_detected:
                break

    # Also check the original parsing error detection
    if not parsing_error_detected:
        for i, ref_author in enumerate(unmatched_ref):
            ref_last = ref_author.get('last_name', '').lower().strip()
            ref_first = ref_author.get('first_name', '').lower().strip()

            for j, dblp_author in enumerate(unmatched_dblp):
                dblp_last = dblp_author.get('last_name', '').lower().strip()
                dblp_first = dblp_author.get('first_name', '').lower().strip()

                # Check if ref's last_name matches dblp's first_name (or vice versa)
                # This indicates names are mixed up/shifted
                if ref_last and dblp_first and ref_last == dblp_first:
                    parsing_error_detected = True
                    break
                if ref_first and dblp_last and ref_first == dblp_last:
                    parsing_error_detected = True
                    break

            if parsing_error_detected:
                break
    
    # If parsing error detected, mark all unmatched authors as parsing error
    if parsing_error_detected:
        result['error_classifications'].append('parsing_error')
        result['mismatches'].append(
            'Parsing error detected: author names appear to be shifted or mixed up'
        )
        return result
    
    # Classify errors for unmatched reference authors
    for ref_author in unmatched_ref:
        best_match = None
        best_match_idx = None
        best_match_type = None
        
        for j, dblp_author in enumerate(dblp_subset):
            if j in matched_dblp_indices:
                continue
            
            # Check different types of mismatches
            # Extract first_name and last_name from normalized author dictionaries
            # These are created by normalize_author_name() which uses nameparser
            # Location: ref_author['first_name'] and ref_author['last_name'] 
            # (normalized from the original author string)
            ref_last = ref_author.get('last_name', '').strip()
            dblp_last = dblp_author.get('last_name', '').strip()
            
            ref_first = ref_author.get('first_name', '').strip()
            dblp_first = dblp_author.get('first_name', '').strip()
            
            # Normalize names for comparison (handles accents)
            ref_last_norm = normalize_name_for_comparison(ref_last)
            dblp_last_norm = normalize_name_for_comparison(dblp_last)
            ref_first_norm = normalize_name_for_comparison(ref_first)
            dblp_first_norm = normalize_name_for_comparison(dblp_first)
            
            # Check for compound last names with prefixes (De Choudhury vs Choudhury)
            ref_last_base, ref_last_orig = normalize_last_name_with_prefixes(ref_last)
            dblp_last_base, dblp_last_orig = normalize_last_name_with_prefixes(dblp_last)
            ref_last_base_norm = normalize_name_for_comparison(ref_last_base)
            dblp_last_base_norm = normalize_name_for_comparison(dblp_last_base)
            
            # Last name matches (with accent normalization and prefix handling)
            last_names_match = (ref_last_norm == dblp_last_norm or 
                               ref_last_base_norm == dblp_last_base_norm or
                               names_match_with_accents(ref_last, dblp_last))
            
            if last_names_match:
                # Check if first names match (with accent normalization and middle initial handling)
                first_names_match = (ref_first_norm == dblp_first_norm or 
                                    names_match_with_accents(ref_first, dblp_first))
                
                # Also check for middle initial matches (e.g., "Ed" vs "Ed H.")
                if not first_names_match:
                    first_names_match = handle_middle_initial_match(ref_author, dblp_author)

                # Reuse historical part-based comparison fallback (configurable metric).
                if not first_names_match:
                    first_names_match = names_match_by_parts(
                        ref_author,
                        dblp_author,
                        similarity_method=similarity_method,
                        fuzz_threshold=name_part_fuzz_threshold,
                        damerau_max_distance=name_part_damerau_max_distance,
                    )
                
                if not first_names_match:
                    # Check if first names match by initials (including compound initials like "K.-T" matching "Kwang-Ting")
                    # Use the enhanced initial_matches function that handles compound initials
                    # IMPORTANT: initial_matches now only returns True if one is actually an initial
                    # (single letter) and the other is a full name. It will NOT match different full names
                    # like "Jeff" vs "Jeffrey" or "Alex" vs "Alexander"
                    initials_match = initial_matches(ref_author['first_name'], dblp_author['first_name'])
                    
                    # Also check simple initial match: only if one is a single letter initial
                    ref_first_initial = ref_first_norm.replace(' ', '')
                    dblp_first_initial = dblp_first_norm.replace(' ', '')
                    # Only match if one is a single letter (actual initial) and the other starts with that letter
                    ref_is_single_initial = len(ref_first_initial) == 1
                    dblp_is_single_initial = len(dblp_first_initial) == 1
                    simple_initials_match = False
                    if ref_is_single_initial and not dblp_is_single_initial:
                        # ref is an initial, dblp is full name - match if first letter matches
                        simple_initials_match = ref_first_initial[0] == dblp_first_initial[0]
                    elif dblp_is_single_initial and not ref_is_single_initial:
                        # dblp is an initial, ref is full name - match if first letter matches
                        simple_initials_match = dblp_first_initial[0] == ref_first_initial[0]
                    elif ref_is_single_initial and dblp_is_single_initial:
                        # Both are single letter initials - match if they're the same
                        simple_initials_match = ref_first_initial[0] == dblp_first_initial[0]
                    
                    # If initials match (compound or simple), consider them matched and skip mismatch reporting
                    if initials_match or simple_initials_match:
                        # Mark this DBLP author as matched and break to skip adding as mismatch
                        matched_dblp_indices.add(j)
                        break  # Break out of inner loop, this reference author is matched
                    
                    # Check if it's just an accent difference
                    if ref_first_norm == dblp_first_norm:
                        # Names match after accent normalization - this is not a mismatch
                        matched_dblp_indices.add(j)
                        break
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'first_name_mismatch'
                else:
                    # First names match (with accents/middle initials) - this is a match
                    matched_dblp_indices.add(j)
                    break
                continue
            
            # First name matches but last name differs
            first_names_match = (ref_first_norm == dblp_first_norm or 
                                names_match_with_accents(ref_first, dblp_first))
            
            # Also check for middle initial matches
            if not first_names_match:
                first_names_match = handle_middle_initial_match(ref_author, dblp_author)

            if not first_names_match:
                first_names_match = names_match_by_parts(
                    ref_author,
                    dblp_author,
                    similarity_method=similarity_method,
                    fuzz_threshold=name_part_fuzz_threshold,
                    damerau_max_distance=name_part_damerau_max_distance,
                )
            
            if first_names_match:
                # Check if last names match with accent normalization and prefix handling
                if not last_names_match:
                    # Check if it's just an accent difference
                    if ref_last_norm == dblp_last_norm or ref_last_base_norm == dblp_last_base_norm:
                        # Names match after accent/prefix normalization - this is not a mismatch
                        matched_dblp_indices.add(j)
                        break
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'last_name_mismatch'
                else:
                    # Both first and last names match - this is a match
                    matched_dblp_indices.add(j)
                    break
                continue
            
            # Check if first names match by initials
            ref_first_initial = ref_first_norm.replace(' ', '')
            dblp_first_initial = dblp_first_norm.replace(' ', '')
            first_initials_match = (len(ref_first_initial) == 1 and len(dblp_first_initial) >= 1 and
                                  ref_first_initial[0] == dblp_first_initial[0])
            if first_initials_match:
                if not last_names_match:
                    # Check if it's just an accent/prefix difference
                    if ref_last_norm == dblp_last_norm or ref_last_base_norm == dblp_last_base_norm:
                        # Names match after accent/prefix normalization - this is not a mismatch
                        matched_dblp_indices.add(j)
                        break
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'last_name_mismatch'
                else:
                    # Both match - this is a match
                    matched_dblp_indices.add(j)
                    break
                continue
            
            # Check if names match without accents (but have accents in original)
            if (ref_last_norm == dblp_last_norm and ref_first_norm == dblp_first_norm and
                (ref_last != dblp_last or ref_first != dblp_first)):
                # Names match after normalization - this is not a mismatch
                matched_dblp_indices.add(j)
                break
        
        if best_match:
            matched_dblp_indices.add(best_match_idx)
            result['error_classifications'].append(best_match_type)
            result['mismatches'].append(
                f"{best_match_type}: {ref_author['first_name']} {ref_author['last_name']} vs "
                f"{best_match['first_name']} {best_match['last_name']}"
            )
        else:
            result['error_classifications'].append('author_not_found')
            result['mismatches'].append(
                f"Author not found in DBLP: {ref_author['first_name']} {ref_author['last_name']}"
            )
    
    # Check for order issues - if all reference authors match but in different order
    if len(matches) == len(ref_subset):
        # Check if order is preserved (authors should match in same relative positions)
        # matches contains tuples: (ref_index, dblp_index, ref_author, dblp_author)
        # Sort matches by reference index and check if DBLP indices are also in order
        sorted_matches = sorted(matches, key=lambda x: x[0])
        order_mismatch = False
        for i, (ref_idx, dblp_idx, _, _) in enumerate(sorted_matches):
            # Check if the relative order is preserved
            if i > 0 and dblp_idx < sorted_matches[i-1][1]:
                order_mismatch = True
                break

        if order_mismatch:
            result['error_classifications'].append('author_order_wrong')
            result['mismatches'].append('Authors match but order differs')
            # Don't mark as complete match if order is wrong
            result['matches'] = False
        else:
            # All authors match and order is correct
            result['matches'] = True

    # Heuristic: if there are 3 or more errors, classify as parsing error
    if len(result['error_classifications']) >= 3:
        result['error_classifications'] = ['parsing_error']
        # Update mismatches description to indicate this is likely a parsing error
        result['mismatches'] = [
            f"Multiple errors detected ({len(result['mismatches'])} issues) - likely parsing error: " +
            '; '.join(result['mismatches'][:3]) + ('...' if len(result['mismatches']) > 3 else '')
        ]

    return result


def normalize_author_name(name: str) -> Dict[str, str]:
    """
    Normalize author names into a consistent format using nameparser.
    Removes both 4-digit suffixes and DBLP-style numeric suffixes.
    Detects and handles parsing errors like "*" characters.
    
    This function is copied from citation_pipeline.py to avoid import issues.
    
    Args:
        name (str): Full author name string
        
    Returns:
        Dict[str, str]: Normalized name components with keys:
            - first_name: First name (accessed as author['first_name'])
            - middle_name: Middle name/initial
            - last_name: Last name (accessed as author['last_name'])
            - suffix: Suffix (e.g., Jr., III)
            - title: Title (e.g., Dr., Prof.)
            - original: Original name string
            - parsing_error: Boolean flag indicating if parsing error detected
    
    Note: First and last names are extracted and compared in check_author_with_minimum_lists()
          at lines 146-150 (ref_first, ref_last, dblp_first, dblp_last)
    """
    # Check for invalid characters that indicate parsing errors
    # Common parsing errors: "*", empty strings, semicolons, or names that are just punctuation
    has_parsing_error = False
    original_name = name  # Keep original for error detection
    
    if not name or not name.strip():
        has_parsing_error = True
    # Check if name contains asterisk (common parsing error marker)
    elif '*' in name:
        has_parsing_error = True
    # Check if name is just an asterisk or starts/ends with asterisk
    elif name.strip() == '*' or name.strip().startswith('* ') or name.strip().endswith(' *'):
        has_parsing_error = True
    # Check if name starts with semicolon (parsing error - name fragment)
    elif name.strip().startswith(';'):
        has_parsing_error = True
    # Check if name is a single letter (likely a parsing fragment, not a valid name)
    elif len(name.strip()) == 1 and name.strip().isalpha():
        has_parsing_error = True
    # Check if name is a single non-alphanumeric character
    elif len(name.strip()) == 1 and not name.strip().isalnum():
        has_parsing_error = True
    # Check if name is very short (1-2 characters) and doesn't look like a valid name
    elif len(name.strip()) <= 2 and not any(c.isalpha() for c in name.strip()):
        has_parsing_error = True
    
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)  # Remove trailing numbers like 0001
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)  # Remove internal numbers
    
    # Remove asterisks and other invalid characters for parsing
    cleaned_name = cleaned_name.replace('*', '').strip()
    
    # If after cleaning we have nothing valid, mark as parsing error
    if not cleaned_name:
        has_parsing_error = True
    
    # Parse the cleaned name using nameparser
    parsed = HumanName(cleaned_name)
    
    # Check if parsing resulted in invalid data (e.g., first_name is "*" or empty)
    if parsed.first == '*' or parsed.last == '*' or (not parsed.first and not parsed.last):
        has_parsing_error = True
    
    # Fix common nameparser misparsing: when title is a single word that looks like part of a name,
    # combine it with first_name (e.g., "Se Young Chun" -> title="Se", first="Young" should be first="Se Young")
    # This handles cases like "Se Young", "Pang Wei", etc. where the first part is misclassified as title
    first_name = parsed.first or ''
    middle_name = parsed.middle or ''
    title = parsed.title or ''
    
    # If title is a single word and first_name exists, check if title should be part of first_name
    # This happens when nameparser incorrectly splits multi-word first names
    if title and first_name and ' ' not in title:
        # Combine title with first_name if it looks like part of the name (not a real title like "Dr.")
        # Real titles are usually short abbreviations or honorifics
        real_titles = {'dr', 'mr', 'mrs', 'ms', 'prof', 'professor', 'sir', 'madam', 'lord', 'lady'}
        if title.lower() not in real_titles:
            # Title is likely part of the first name, combine them
            first_name = f"{title} {first_name}".strip()
            title = ''  # Clear title since we've merged it
    
    # Also handle middle names - if we have a multi-word first name pattern, combine middle with first
    # This helps with cases like "Se Young Chun" where middle might be empty but we need "Se Young"
    # Actually, nameparser handles this differently - let's focus on the title issue above
    
    return {
        'first_name': first_name,
        'middle_name': middle_name,
        'last_name': parsed.last or '',
        'suffix': parsed.suffix or '',
        'title': title,
        'original': name,  # Keep original for reference
        'parsing_error': has_parsing_error  # Flag for parsing errors
    }


def find_json_files(input_dir: str, num_files: Optional[int] = None) -> List[str]:
    """
    Find JSON files in the input directory and return a random sample.

    Args:
        input_dir: Root directory containing JSON files
        num_files: Number of files to return (None for all files)

    Returns:
        List of file paths
    """
    json_files = []

    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    # Randomly sample if we have more files than requested
    if num_files is not None and len(json_files) > num_files:
        json_files = random.sample(json_files, num_files)

    logger.info(f"Found {len(json_files)} JSON files to process")
    return json_files


def parse_source_names(value: str) -> List[str]:
    """Parse and validate comma-separated source names from CLI input."""
    raw_names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_names:
        raise argparse.ArgumentTypeError(
            "At least one source is required. Allowed values: "
            + ", ".join(AVAILABLE_SOURCES)
        )

    invalid = sorted({name for name in raw_names if name not in AVAILABLE_SOURCES})
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown source(s): {', '.join(invalid)}. "
            f"Allowed values: {', '.join(AVAILABLE_SOURCES)}"
        )

    deduped: List[str] = []
    seen = set()
    for name in raw_names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def parse_similarity_method(value: str) -> str:
    """
    Parse similarity method with support for ``fuzz.ratio`` alias.
    """
    normalized = (value or "").strip().lower()
    if normalized in {"fuzz", "fuzz.ratio"}:
        return "fuzz"
    if normalized in {"damerau", "damerau_levenshtein", "damerau-levenshtein"}:
        return "damerau"
    raise argparse.ArgumentTypeError(
        f"Unknown similarity method: {value}. "
        f"Allowed: {', '.join(SIMILARITY_METHODS)} (and alias: fuzz.ratio)"
    )


def initialize_sources(source_names: List[str], dblp_xml: str) -> List[CitationSource]:
    """Initialize configured citation sources."""
    sources: List[CitationSource] = []

    if "dblp" in source_names:
        if not os.path.exists(dblp_xml):
            raise FileNotFoundError(f"DBLP XML file not found: {dblp_xml}")

        logger.info("Initializing DBLP parser from: %s", dblp_xml)
        dblp_parser = DblpParser(
            xml_path=dblp_xml,
            cache_dir="dblp_cache",
            index_name="dblp_index",
        )
        sources.append(DBLPSource(dblp_parser))

    if "acl" in source_names:
        sources.append(ACLAnthologySource())

    if "arxiv" in source_names:
        sources.append(ArXivSource())

    return sources


def find_best_database_match(
    title: str,
    sources: List[CitationSource],
    threshold: float = 5.0,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
) -> tuple[Optional[SourceMatch], float]:
    """Find the best title match across the configured sources."""
    best_match: Optional[SourceMatch] = None
    best_similarity = 0.0

    for source in sources:
        candidate = source.search_by_title(title, threshold=threshold)
        if not candidate:
            continue

        similarity = calculate_title_similarity(
            title,
            candidate.title,
            similarity_method=similarity_method,
        )
        if best_match is None or similarity > best_similarity:
            best_match = candidate
            best_similarity = similarity

    return best_match, best_similarity


def validate_reference(
    reference: Dict[str, Any],
    sources: List[CitationSource],
    threshold: float = 5.0,
    title_similarity_threshold: float = 95.0,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
    name_part_fuzz_threshold: float = DEFAULT_NAME_PART_FUZZ_THRESHOLD,
    name_part_damerau_max_distance: int = DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
) -> Dict[str, Any]:
    """
    Validate a single reference by querying configured sources and comparing authors.
    
    Args:
        reference: Reference dictionary with 'title' and 'authors' keys
        sources: Initialized source instances to query for title matches
        threshold: Source-specific retrieval threshold (DBLP BM25, optional elsewhere)
        title_similarity_threshold: Minimum string similarity (0-100) between titles to consider match
        
    Returns:
        Dictionary with validation results:
            - reference: Original reference data
            - dblp_match: Legacy field containing matched publication data
            - matched_source: Name of selected source ('dblp', 'acl', or 'arxiv')
            - authors_match: Boolean indicating if authors match
            - mismatches: List of mismatch descriptions
            - error_classifications: List of error types (first_name_mismatch, last_name_mismatch, etc.)
            - title_similarity: Similarity score between titles
            - validation_status: 'matched', 'no_database_match', 'title_mismatch', 'author_mismatch', or 'error'
    """
    result = {
        'reference': reference,
        'dblp_match': None,
        'matched_source': None,
        'authors_match': False,
        'mismatches': [],
        'error_classifications': [],
        'title_similarity': 0.0,
        'validation_status': 'unknown'
    }
    
    title = reference.get('title', '')
    if not title:
        result['validation_status'] = 'error'
        result['mismatches'].append('No title in reference')
        return result

    # Skip validation for non-academic references like Wikipedia
    title_lower = title.lower()
    if 'wikipedia' in title_lower or title_lower.strip() == 'wikipedia':
        result['validation_status'] = 'skipped'
        result['mismatches'].append('Skipped validation for non-academic reference (Wikipedia)')
        return result
    
    # Query configured sources for the best title match
    try:
        best_match, title_similarity = find_best_database_match(
            title,
            sources,
            threshold=threshold,
            similarity_method=similarity_method,
        )
        result['title_similarity'] = title_similarity

        if not best_match:
            result['validation_status'] = 'no_database_match'
            result['mismatches'].append(f'No database match found for title: {title[:100]}')
            return result

        matched_title = best_match.title

        # Only consider if title similarity is >= threshold
        if title_similarity < title_similarity_threshold:
            result['validation_status'] = 'title_mismatch'
            result['mismatches'].append(
                f'Title similarity too low: {title_similarity:.1f}% '
                f'(threshold: {title_similarity_threshold}%). '
                f'Reference: "{title[:100]}", Matched ({best_match.source}): "{matched_title[:100]}"'
            )
            return result

        result['dblp_match'] = {
            'source': best_match.source,
            'title': matched_title,
            'authors': best_match.authors,
            'year': best_match.year,
            'venue': best_match.venue,
            'metadata': best_match.metadata,
        }
        result['matched_source'] = best_match.source
        
        # Normalize authors from reference
        ref_authors_raw = reference.get('authors', [])
        # Handle case where authors is a string instead of a list
        if isinstance(ref_authors_raw, str):
            ref_authors_raw = [ref_authors_raw]
        ref_authors_normalized = [normalize_author_name(author) for author in ref_authors_raw]
        
        # Normalize authors from selected source
        matched_authors_normalized = [normalize_author_name(author) for author in best_match.authors]
        
        # Check if authors match using first 10 authors comparison
        author_check_result = check_author_with_minimum_lists(
            ref_authors_normalized,
            matched_authors_normalized,
            title,
            max_authors=10,
            similarity_method=similarity_method,
            name_part_fuzz_threshold=name_part_fuzz_threshold,
            name_part_damerau_max_distance=name_part_damerau_max_distance,
        )
        
        result['authors_match'] = author_check_result['matches']
        result['mismatches'] = author_check_result['mismatches']
        result['error_classifications'] = author_check_result['error_classifications']
        
        if author_check_result['matches']:
            result['validation_status'] = 'matched'
        else:
            result['validation_status'] = 'author_mismatch'
            
    except Exception as e:
        logger.error(f"Error validating reference '{title[:50]}...': {e}")
        result['validation_status'] = 'error'
        result['mismatches'].append(f'Error during validation: {str(e)}')
    
    return result


def clean_validation_result(result: Dict[str, Any], source_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up validation result by adding source info and removing specified fields.
    
    Args:
        result: Validation result dictionary
        source_info: Information about the source paper
        
    Returns:
        Cleaned result dictionary
    """
    # Add source info
    result['source_paper'] = source_info
    
    # Remove fields from reference
    if 'reference' in result:
        ref = result['reference']
        for field in ['publication_date', 'year', 'urls', 'target']:
            ref.pop(field, None)
            
    # Remove fields from dblp_match
    if result.get('dblp_match'):
        dblp = result['dblp_match']
        for field in ['year', 'venue', 'title']:
            dblp.pop(field, None)
            
    # Remove top level fields
    for field in ['title_similarity', 'authors_match']:
        result.pop(field, None)
        
    return result


def process_json_file(
    json_path: str,
    sources: List[CitationSource],
    threshold: float = 5.0,
    title_similarity_threshold: float = 95.0,
    similarity_method: str = DEFAULT_SIMILARITY_METHOD,
    name_part_fuzz_threshold: float = DEFAULT_NAME_PART_FUZZ_THRESHOLD,
    name_part_damerau_max_distance: int = DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
) -> Dict[str, Any]:
    """
    Process a single JSON file and validate all its references.
    
    Args:
        json_path: Path to JSON file
        sources: Initialized citation lookup sources
        threshold: Source-specific retrieval threshold (DBLP BM25, optional elsewhere)
        similarity_method: String similarity method (``fuzz`` or ``damerau``)
        
    Returns:
        Dictionary with file processing results:
            - file_path: Path to processed file
            - references_count: Total number of references
            - validated_count: Number of references successfully validated
            - matched_count: Number of references with matching authors
            - mismatch_count: Number of references with author mismatches
            - no_match_count: Number of references without usable database match
            - error_count: Number of references with errors
            - results: List of validation results for each reference
    """
    logger.info(f"Processing file: {json_path}")
    
    file_result = {
        'file_path': json_path,
        'references_count': 0,
        'validated_count': 0,
        'matched_count': 0,
        'mismatch_count': 0,
        'no_match_count': 0,
        'error_count': 0,
        'skipped_count': 0,
        'results': []
    }
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        references = data.get('references', [])
        file_result['references_count'] = len(references)
        
        # Extract source paper info
        source_info = data.get('biblio', {})
        # Add file path to source info
        source_info['file_path'] = json_path
        
        logger.info(f"Found {len(references)} references in {json_path}")
        
        # Validate each reference
        total_refs = len(references)
        for ref_idx, ref in enumerate(references, start=1):
            if ref_idx == 1 or ref_idx % 10 == 0 or ref_idx == total_refs:
                logger.info(
                    "  reference %s/%s in %s",
                    ref_idx,
                    total_refs,
                    Path(json_path).name,
                )
            validation_result = validate_reference(
                ref,
                sources,
                threshold,
                title_similarity_threshold,
                similarity_method=similarity_method,
                name_part_fuzz_threshold=name_part_fuzz_threshold,
                name_part_damerau_max_distance=name_part_damerau_max_distance,
            )
            
            # Clean result (adds source info and removes unwanted fields)
            cleaned_result = clean_validation_result(validation_result, source_info)
            file_result['results'].append(cleaned_result)
            
            # Update counters
            status = validation_result['validation_status']
            if status == 'matched':
                file_result['matched_count'] += 1
                file_result['validated_count'] += 1
            elif status == 'author_mismatch':
                file_result['mismatch_count'] += 1
                file_result['validated_count'] += 1
            elif status == 'title_mismatch':
                file_result['no_match_count'] += 1
            elif status in {'no_dblp_match', 'no_database_match'}:
                file_result['no_match_count'] += 1
            elif status == 'error':
                file_result['error_count'] += 1
            elif status == 'skipped':
                file_result['skipped_count'] += 1
        
        logger.info(f"Completed {json_path}: {file_result['matched_count']} matched, "
                   f"{file_result['mismatch_count']} mismatches, "
                   f"{file_result['no_match_count']} no database match")
        
    except Exception as e:
        logger.error(f"Error processing file {json_path}: {e}")
        file_result['error_count'] = 1
    
    return file_result


def main():
    """Main function to run citation validation."""
    parser = argparse.ArgumentParser(
        description='Validate citation authors against selected reference databases.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default='data/parsed_jsons',
                       help='Directory containing parsed JSON files')
    parser.add_argument('--dblp-xml', type=str, default='data/dblp.xml',
                       help='Path to DBLP XML file (required when "dblp" source is enabled)')
    parser.add_argument(
        '--sources',
        type=str,
        default=DEFAULT_SOURCES,
        help=(
            'Comma-separated sources to query. '
            f'Allowed: {", ".join(AVAILABLE_SOURCES)}'
        ),
    )
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for validation results (default: validation_results)')
    parser.add_argument('--num-files', type=int, default=None,
                       help='Number of JSON files to process (default: all files)')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='BM25 score threshold for DBLP title matching')
    parser.add_argument('--title-similarity-threshold', type=float, default=95.0,
                       help='Minimum string similarity (0-100) between titles to consider match')
    parser.add_argument(
        '--similarity-method',
        type=str,
        default=DEFAULT_SIMILARITY_METHOD,
        help='Similarity backend: fuzz, fuzz.ratio, or damerau',
    )
    parser.add_argument(
        '--name-part-fuzz-threshold',
        type=float,
        default=DEFAULT_NAME_PART_FUZZ_THRESHOLD,
        help='Part-level fuzz threshold (0-100) used when similarity-method=fuzz',
    )
    parser.add_argument(
        '--name-part-damerau-max-distance',
        type=int,
        default=DEFAULT_NAME_PART_DAMERAU_MAX_DISTANCE,
        help='Part-level Damerau distance threshold used when similarity-method=damerau',
    )
    
    args = parser.parse_args()
    try:
        args.sources = parse_source_names(args.sources)
        args.similarity_method = parse_similarity_method(args.similarity_method)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.name_part_damerau_max_distance < 0:
        parser.error("--name-part-damerau-max-distance must be >= 0")
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")

    try:
        sources = initialize_sources(args.sources, args.dblp_xml)
    except Exception as e:
        logger.error(f"Failed to initialize sources ({', '.join(args.sources)}): {e}")
        return

    if not sources:
        logger.error("No sources initialized. Nothing to validate.")
        return

    logger.info("Using sources: %s", ", ".join(source.source_name for source in sources))
    logger.info("Using similarity method: %s", args.similarity_method)
    
    # Find JSON files to process
    json_files = find_json_files(args.input_dir, args.num_files)
    
    if not json_files:
        logger.warning("No JSON files found to process")
        return
    logger.info("Starting validation for %s files", len(json_files))
    
    # Process each JSON file
    all_results = []
    total_stats = {
        'files_processed': 0,
        'total_references': 0,
        'total_matched': 0,
        'total_mismatches': 0,
        'total_no_match': 0,
        'total_errors': 0,
        'total_skipped': 0
    }
    
    progress = tqdm(
        json_files,
        desc="Processing files",
        disable=not sys.stderr.isatty(),
    )
    for idx, json_file in enumerate(progress, start=1):
        logger.info("[%s/%s] validating %s", idx, len(json_files), json_file)
        file_result = process_json_file(
            json_file,
            sources,
            args.threshold,
            args.title_similarity_threshold,
            similarity_method=args.similarity_method,
            name_part_fuzz_threshold=args.name_part_fuzz_threshold,
            name_part_damerau_max_distance=args.name_part_damerau_max_distance,
        )
        all_results.append(file_result)

        # Update total statistics
        total_stats['files_processed'] += 1
        total_stats['total_references'] += file_result['references_count']
        total_stats['total_matched'] += file_result['matched_count']
        total_stats['total_mismatches'] += file_result['mismatch_count']
        total_stats['total_no_match'] += file_result['no_match_count']
        total_stats['total_errors'] += file_result['error_count']
        total_stats['total_skipped'] += file_result['skipped_count']
        progress.set_postfix(
            matched=total_stats['total_matched'],
            mismatches=total_stats['total_mismatches'],
            no_match=total_stats['total_no_match'],
        )
    
    # Extract all validation results for analysis
    all_validation_results = []
    for file_data in all_results:
        all_validation_results.extend(file_data.get('results', []))
    
    # Separate mismatches and matches
    mismatches = []
    matches = []
    
    for result in all_validation_results:
        if result['validation_status'] == 'author_mismatch':
            mismatches.append(result)
        elif result['validation_status'] == 'matched':
            matches.append(result)

    # Sort mismatches to put parsing errors at the bottom
    def has_parsing_error(result):
        return 'parsing_error' in result.get('error_classifications', [])

    mismatches.sort(key=has_parsing_error)
    
    # Perform analysis on results
    error_counts = Counter()
    title_similarities = []
    
    for result in all_validation_results:
        # Count error classifications
        for error_type in result.get('error_classifications', []):
            error_counts[error_type] += 1
        
        # Collect title similarities
        similarity = result.get('title_similarity', 0.0)
        if similarity > 0:
            title_similarities.append(similarity)
    
    # Calculate title similarity statistics
    title_sim_stats = {}
    if title_similarities:
        title_similarities.sort()
        n = len(title_similarities)
        title_sim_stats = {
            'count': n,
            'min': min(title_similarities),
            'max': max(title_similarities),
            'mean': sum(title_similarities) / n,
            'median': title_similarities[n // 2]
        }
    
    # Create final output structure with integrated analysis
    # Put mismatches on top, matches at bottom
    output_data = {
        'summary': total_stats,
        'sources': args.sources,
        'similarity': {
            'method': args.similarity_method,
            'name_part_fuzz_threshold': args.name_part_fuzz_threshold,
            'name_part_damerau_max_distance': args.name_part_damerau_max_distance,
        },
        'analysis': {
            'error_classifications': dict(error_counts),
            'title_similarity_stats': title_sim_stats,
            'mismatch_count': len(mismatches),
            'match_count': len(matches)
        },
        'mismatches': mismatches,
        'matches': matches,
        'files': all_results
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write main results JSON file
    main_json_path = os.path.join(args.output_dir, 'validation_results.json')
    try:
        with open(main_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Main results written to: {main_json_path}")
    except Exception as e:
        logger.error(f"Error writing results to file: {e}")
        return
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Files processed: {total_stats['files_processed']}")
    logger.info(f"Total references: {total_stats['total_references']}")
    logger.info(f"Matched (correct citations): {total_stats['total_matched']}")
    logger.info(f"Mismatches (incorrect citations): {total_stats['total_mismatches']}")
    logger.info(f"No database match / Title mismatch: {total_stats['total_no_match']}")
    logger.info(f"Skipped (non-academic): {total_stats['total_skipped']}")
    logger.info(f"Errors: {total_stats['total_errors']}")
    logger.info("="*60)
    
    # Legacy behavior: write only one output JSON file.


def extract_all_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all validation results from the output data structure.
    
    Args:
        data: The output data dictionary from validation
        
    Returns:
        List of all validation result dictionaries
    """
    all_results = []
    
    # Extract from mismatches and matches sections
    all_results.extend(data.get('mismatches', []))
    all_results.extend(data.get('matches', []))
    
    # Also extract from files section
    for file_data in data.get('files', []):
        all_results.extend(file_data.get('results', []))
    
    # Remove duplicates based on reference ID and title
    seen = set()
    unique_results = []
    for result in all_results:
        if isinstance(result, dict):
            ref = result.get('reference', {})
            key = (ref.get('id', ''), ref.get('title', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
    
    return unique_results


def categorize_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize validation results into different groups."""
    categories = defaultdict(list)
    
    for result in results:
        status = result.get('validation_status', 'unknown')
        classifications = result.get('error_classifications', [])
        
        if status == 'matched':
            categories['matched'].append(result)
        if 'parsing_error' in classifications:
            categories['parsing_errors'].append(result)
        if 'first_name_mismatch' in classifications:
            categories['first_names'].append(result)
        if 'last_name_mismatch' in classifications:
            categories['last_names'].append(result)
        if 'accents_missing' in classifications:
            categories['accents_missing'].append(result)
        if 'author_not_found' in classifications:
            categories['author_not_found'].append(result)
        if 'author_order_wrong' in classifications:
            categories['author_order_wrong'].append(result)
        if 'empty_list' in classifications:
            categories['empty_list'].append(result)
        if status == 'title_mismatch':
            categories['title_mismatches'].append(result)
        if status in {'no_dblp_match', 'no_database_match'}:
            categories['no_database_match'].append(result)
        if status == 'no_dblp_match':
            categories['no_dblp_match'].append(result)
        if status == 'error':
            categories['errors'].append(result)
        if status == 'skipped':
            categories['skipped'].append(result)
    
    categories['summary'] = [{
        'total_results': len(results),
        'categories': {cat: len(results) for cat, results in categories.items() if cat != 'summary'}
    }]
    
    return dict(categories)


def reorganize_results(data: Dict[str, Any], output_dir: str) -> None:
    """Reorganize validation results into categorized files."""
    all_results = extract_all_results(data)
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    categories = categorize_results(all_results)
    
    # Write category files with progress bar
    for category_name, results in tqdm(categories.items(), desc="Writing result files"):
        if category_name == 'summary':
            filename = 'summary.json'
        else:
            filename = f'{category_name}.json'

        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create README
    readme_path = os.path.join(results_dir, 'README.md')
    readme_content = """# Citation Validation Results

This folder contains citation validation results organized by category.

## File Structure

- `matched.json` - Citations that matched correctly with the selected sources
- `parsing_errors.json` - Citations with parsing errors in author names
- `first_names.json` - Citations with first name mismatches
- `last_names.json` - Citations with last name mismatches
- `accents_missing.json` - Citations with missing accents/diacritics
- `author_not_found.json` - Citations where authors were not found in matched source records
- `author_order_wrong.json` - Citations with correct authors but wrong order
- `empty_list.json` - Citations with empty author lists
- `title_mismatches.json` - Citations with title similarity below threshold
- `no_database_match.json` - Citations not found in any configured database
- `no_dblp_match.json` - Legacy category for DBLP-only unmatched results
- `errors.json` - Citations that caused processing errors
- `skipped.json` - Citations that were skipped
- `summary.json` - Summary statistics for all categories

## Statistics

"""
    
    summary = categories.get('summary', [{}])[0]
    if 'categories' in summary:
        readme_content += "| Category | Count |\n|----------|-------|\n"
        for cat, count in sorted(summary['categories'].items()):
            if cat != 'summary':
                readme_content += f"| {cat} | {count} |\n"
        readme_content += f"\n**Total Results:** {summary.get('total_results', 0)}\n"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Successfully reorganized results into: {results_dir}")


if __name__ == '__main__':
    main()
