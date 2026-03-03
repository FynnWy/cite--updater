"""
Author name matching and analysis tool.

This script analyzes differences between parsed and matched author names in academic papers,
focusing on detecting parsing errors, name mismatches, and first name variations.
It handles various name formats including initials, compound names, and reversed name orders.
"""

import json
import os
import logging
from typing import List, Dict, Set, Optional
from difflib import get_close_matches
from rapidfuzz.distance import DamerauLevenshtein
import argparse
from unidecode import unidecode

def setup_logging(output_dir: str) -> None:
    """
    Configure logging to write to both file and console.
    
    Args:
        output_dir: Directory where log file will be stored
    """
    log_file = os.path.join(output_dir, 'author_analysis.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def detect_parsing_error(name: Dict[str, str], paper_title: str) -> Optional[str]:
    """
    Check for critical parsing errors in author names.
    Only detects serious issues like duplicate first/last names.
    
    Args:
        name: Dictionary containing author name components
        paper_title: Title of the paper for context in error messages
        
    Returns:
        str: Error message if critical parsing error detected, None otherwise
    """
    # Only check for duplicate first/last name as it indicates a serious parsing issue
    if name['first_name'].lower() == name['last_name'].lower():
        return f"Duplicate first/last name: {name['first_name']}"
    
    return None

def normalize_name(name: Dict[str, str]) -> str:
    """
    Convert author dict to normalized string for comparison.
    Uses first and last name only, ignoring the original field.
    
    Args:
        name: Dictionary containing author name components
        
    Returns:
        str: Normalized name string in lowercase
    """
    return f"{name['first_name']} {name['last_name']}".lower()

def find_closest_match(name: str, name_list: List[str]) -> Optional[tuple[str, float]]:
    """Find closest matching name from a list using difflib."""
    matches = get_close_matches(name, name_list, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

def get_initials(name: str) -> str:
    """
    Extract initials from a name string, handling multiple words, hyphens, and accents.
    Handles patterns like "K.-T" (compound initials) and "Kwang-Ting" (compound names).
    """
    initials = []
    # First, handle compound initials like "K.-T" by splitting on periods and hyphens
    # Replace periods and hyphens with spaces, then split
    normalized = name.replace('.', ' ').replace('-', ' ').strip()
    for word in normalized.split():
        cleaned = word.strip()
        if cleaned:
            # Normalize accents before taking first character
            normalized_first = unidecode(cleaned[0].lower())
            initials.append(normalized_first)
    return ''.join(initials)

def is_initial(name: str) -> bool:
    """Check if a name component is an initial (single letter possibly with period)."""
    return len(name.replace('.', '').strip()) == 1

def is_compound_initial(name: str) -> bool:
    """
    Check if a name component is a compound initial pattern like "K.-T", "X.-S", "C.-J".
    These are initials separated by hyphens and periods.
    """
    # Remove periods and hyphens, check if all remaining characters are single letters
    cleaned = name.replace('.', '').replace('-', '').strip()
    # Check if it's a pattern like "KT" (2-3 letters) or matches compound initial pattern
    if len(cleaned) <= 3 and cleaned.isalpha():
        # Check if original has the pattern of initials separated by punctuation
        if '-' in name or ('.' in name and len(name.replace('.', '').replace('-', '').strip()) <= 3):
            return True
    return False

def normalize_compound_name(name: str) -> str:
    """Normalize compound names by removing spaces and hyphens."""
    return ''.join(c.lower() for c in name if c.isalnum())

def initial_matches(name1: str, name2: str) -> bool:
    """
    Check if names match by initials, handling multiple words.
    IMPORTANT: Only matches when one is actually an initial (single letter) and the other is a full name.
    Does NOT match different full names like "Jeff" vs "Jeffrey" or "Alex" vs "Alexander".
    
    Handles cases like:
    - "S" matching "Se Young" (single initial matches first initial of multi-word name)
    - "K.-T" matching "Kwang-Ting" (compound initials)
    - "A B" matching "Alexander Benjamin" (multiple initials)
    
    Does NOT match:
    - "Jeff" vs "Jeffrey" (both are full names, not initials)
    - "Alex" vs "Alexander" (both are full names, not initials)
    """
    # Check if name1 is actually an initial (single letter, possibly with period)
    name1_is_initial = is_initial(name1) or is_compound_initial(name1)
    # Check if name2 is actually an initial (single letter, possibly with period)
    name2_is_initial = is_initial(name2) or is_compound_initial(name2)
    
    # If both are full names (not initials), don't match them even if they share initials
    # This prevents matching "Jeff" with "Jeffrey" or "Alex" with "Alexander"
    if not name1_is_initial and not name2_is_initial:
        return False
    
    init1 = get_initials(name1)
    init2 = get_initials(name2)
    
    if not init1 or not init2:
        return False
    
    # If one is a single initial and the other is multi-character, check if single initial
    # matches the first character of the multi-character initial
    if len(init1) == 1 and len(init2) > 1:
        return init1[0] == init2[0]
    elif len(init2) == 1 and len(init1) > 1:
        return init2[0] == init1[0]
    
    # Both are initials (single or compound): check if they match
    # For compound initials like "K.-T" vs "Kwang-Ting", check if initials match
    if name1_is_initial and name2_is_initial:
        return init1 == init2 or (len(init1) == 1 and init1[0] == init2[0]) or (len(init2) == 1 and init2[0] == init1[0])
    
    # One is an initial and the other is a full name: check if initial matches first letter(s)
    if name1_is_initial and not name2_is_initial:
        return init1[0] == init2[0] if init1 and init2 else False
    elif name2_is_initial and not name1_is_initial:
        return init2[0] == init1[0] if init1 and init2 else False
    
    return False

def is_name_match(name1: Dict[str, str], name2: Dict[str, str]) -> bool:
    """
    Enhanced name matching that handles:
    - Initials (D. == David)
    - Reversed names (Maria Edoardo == Edoardo Maria)
    - Middle names
    - Combined first names
    """
    def normalize_component(text: str) -> str:
        return text.lower().replace('.', '').replace('-', '').strip()
    
    # Get all name components
    n1_first = normalize_component(name1['first_name'])
    n1_middle = normalize_component(name1['middle_name'])
    n1_last = normalize_component(name1['last_name'])
    n2_first = normalize_component(name2['first_name'])
    n2_middle = normalize_component(name2['middle_name'])
    n2_last = normalize_component(name2['last_name'])
    
    # Combine all parts for each name
    n1_parts = [p for p in [n1_first, n1_middle, n1_last] if p]
    n2_parts = [p for p in [n2_first, n2_middle, n2_last] if p]
    
    # Check for exact last name match
    last_match = n1_last == n2_last
    
    # Improve reversed name detection
    def get_all_name_parts(name: Dict[str, str]) -> List[str]:
        parts = []
        if name['first_name']: parts.extend(name['first_name'].lower().split())
        if name['middle_name']: parts.extend(name['middle_name'].lower().split())
        if name['last_name']: parts.extend(name['last_name'].lower().split())
        return parts
    
    # Check if all parts match regardless of order
    n1_all_parts = set(get_all_name_parts(name1))
    n2_all_parts = set(get_all_name_parts(name2))
    if n1_all_parts and n2_all_parts and n1_all_parts == n2_all_parts:
        return True
    
    # Check various name matching scenarios
    if last_match:
        # Case 1: Direct first name match
        if n1_first == n2_first:
            return True
            
        # Case 2: Initial matches full name (including compound initials like "K.-T" matching "Kwang-Ting")
        # Use initial_matches which checks if one is actually an initial
        # This handles "S" matching "Swami" correctly
        if initial_matches(name1['first_name'], name2['first_name']):
            return True
                
        # Case 3: Reversed first/middle names
        if n1_middle and n2_first == n1_middle:
            return True
        if n2_middle and n1_first == n2_middle:
            return True
            
        # Case 4: See if full names are same without spaces
        full_name1 = ''.join(n1_parts)
        full_name2 = ''.join(n2_parts)
        if full_name1 == full_name2:
            return True
    
    # Check reversed full names
    reversed_match = (n1_first == n2_last and n1_last == n2_first)
    if reversed_match:
        return True
        
    # Handle special case of multiple first names split differently
    full_name1 = ' '.join(n1_parts)
    full_name2 = ' '.join(n2_parts)
    if full_name1 == full_name2:
        return True
        
    # Add compound name handling
    n1_compound = normalize_compound_name(f"{n1_first} {n1_middle} {n1_last}")
    n2_compound = normalize_compound_name(f"{n2_first} {n2_middle} {n2_last}")
    if n1_compound == n2_compound:
        return True
    
    return False

def check_author_lists(parsed_authors: List[Dict[str, str]], 
                      matched_authors: List[Dict[str, str]],
                      paper_title: str) -> List[str]:
    """
    Check author lists with enhanced name matching.
    Focuses on actual name differences rather than parsing variations.
    """
    mismatches = []
    
    # First check for critical parsing errors only
    parsing_errors = []
    for author in parsed_authors:
        if error := detect_parsing_error(author, paper_title):
            parsing_errors.append(error)
    for author in matched_authors:
        if error := detect_parsing_error(author, paper_title):
            parsing_errors.append(error)
            
    if parsing_errors:
        mismatches.append("Critical parsing errors detected:")
        mismatches.extend(f"  {error}" for error in parsing_errors)
    
    # Try to match each parsed author with a matched author
    unmatched_parsed = []
    unmatched_matched = list(matched_authors)
    
    for parsed_author in parsed_authors:
        found_match = False
        for matched_author in unmatched_matched:
            if is_name_match(parsed_author, matched_author):
                unmatched_matched.remove(matched_author)
                found_match = True
                break
        if not found_match:
            unmatched_parsed.append(parsed_author)
    
    if unmatched_parsed or unmatched_matched:
        mismatches.append("Author lists differ:")
        
        for author in unmatched_parsed:
            closest = None
            match_reason = None
            
            for matched_author in unmatched_matched:
                # Try different matching strategies
                if is_name_match(author, matched_author):
                    closest = matched_author
                    match_reason = "names match with different ordering"
                    break
                elif initial_matches(author['first_name'], matched_author['first_name']):
                    closest = matched_author
                    match_reason = "matching initials"
                    break
            
            if closest:
                mismatches.append(
                    f"  Parsed: {author['first_name']} {author['last_name']} ≈ "
                    f"Matched: {closest['first_name']} {closest['last_name']} ({match_reason})"
                )
                unmatched_matched.remove(closest)
            else:
                mismatches.append(f"  Only in parsed: {author['first_name']} {author['last_name']}")
        
        # Report remaining unmatched matched authors
        for author in unmatched_matched:
            mismatches.append(f"  Only in matched: {author['first_name']} {author['last_name']}")
    
    return mismatches

def normalize_text(text: str) -> str:
    """
    Normalize text by:
    1. Converting to lowercase
    2. Removing diacritics
    3. Keeping only alphabetic characters
    4. Splitting hyphenated parts for comparison
    """
    # Convert to lowercase and normalize unicode
    text = text.lower()
    text = unidecode(text)
    # Split by hyphen and compare parts
    parts = text.replace('-', ' ').split()
    # Keep only alphabetic characters in each part
    return [''.join(c for c in part if c.isalpha()) for part in parts]

def parts_are_similar(part1: str, part2: str, max_distance: int = 1) -> bool:
    """
    Check if two name parts are similar using DamerauLevenshtein distance.
    
    Args:
        part1: First name part
        part2: Second name part
        max_distance: Maximum edit distance to consider parts similar
        
    Returns:
        bool: True if parts are similar enough
    """
    # Skip very short names or names with big length difference
    if len(part1) <= 1 or len(part2) <= 1:
        return part1 == part2
    if abs(len(part1) - len(part2)) > max_distance:
        return False
    
    return DamerauLevenshtein.distance(part1, part2) <= max_distance

def analyze_first_name_differences(parsed_authors: List[Dict[str, str]], 
                                 matched_authors: List[Dict[str, str]],
                                 paper_title: str) -> List[str]:
    """
    Analyze differences in first names that:
    1. Are actually different names (not just variations caught by is_name_match)
    2. Have meaningful differences (not just diacritics/hyphenation/minor misspellings)
    3. Only compare first names when last names match
    """
    mismatches = []

    if len(parsed_authors) != len(matched_authors):
        mismatches.append(f"Number of parsed authors ({len(parsed_authors)}) does not match number of matched authors ({len(matched_authors)})")
        return mismatches
    
    for i, parsed_author in enumerate(parsed_authors):
        matched_author = matched_authors[i]
        
        # Skip if names already match according to flexible matching
        if is_name_match(parsed_author, matched_author):
            continue
            
        # Only compare first names if last names match
        parsed_last = parsed_author['last_name'].lower()
        matched_last = matched_author['last_name'].lower()
        
        if parsed_last != matched_last:
            continue
            
        # Get normalized first names as lists of parts
        parsed_parts = normalize_text(parsed_author['first_name'])
        matched_parts = normalize_text(matched_author['first_name'])
        
        # Skip if any parts are similar (exact match or edit distance 1)
        if any(parts_are_similar(p1, p2) for p1 in parsed_parts for p2 in matched_parts):
            continue
            
        # Only report if the names are substantially different
        if any(len(p) > 1 for p in parsed_parts) and any(len(p) > 1 for p in matched_parts):
            # Check if first letters match in any parts
            parsed_firsts = {p[0] for p in parsed_parts if p}
            matched_firsts = {p[0] for p in matched_parts if p}
            if not parsed_firsts.intersection(matched_firsts):
                continue
                
            mismatch = (f"Name mismatch: {parsed_author['first_name']} {parsed_last} vs "
                       f"{matched_author['first_name']} {matched_last}")
            mismatches.append(mismatch)
    
    return mismatches

def analyze_author_matches(input_file: str, output_dir: str) -> None:
    """
    Main function to analyze author name matches and differences.
    
    Args:
        input_file: Path to the input JSON file containing author matches
        output_dir: Directory where output files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Load the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in '{input_file}'.")
        return

    # Process each entry
    first_name_results = []
    mismatch_results = []
    
    logging.info(f"Processing author matches from {input_file}")
    
    for entry in data:
        if not entry.get('matched_authors'):
            continue
            
        # Analyze first name differences for entries with single mismatch
        if len(entry.get('mismatches', [])) == 1:
            mismatches = analyze_first_name_differences(
                entry['parsed_authors'],
                entry['matched_authors'],
                entry['title']
            )
            if mismatches:
                first_name_results.append({
                    'title': entry['title'],
                    'mismatches': mismatches
                })
        
        # Check for general author list mismatches
        author_mismatches = check_author_lists(
            entry['parsed_authors'],
            entry['matched_authors'],
            entry['title']
        )
        if author_mismatches:
            mismatch_results.append({
                'title': entry['title'],
                'mismatches': author_mismatches
            })

    # Write results to output files
    output_files = {
        'first_name_differences.json': first_name_results,
        'author_mismatches.json': mismatch_results
    }
    
    for filename, results in output_files.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Wrote results to: {output_path}")
        
        # Log summary
        if results:
            logging.info(f"Found in {filename}:")
            for result in results:
                logging.info(f"\nTitle: {result['title']}")
                for mismatch in result['mismatches']:
                    logging.info(f"  {mismatch}")
        else:
            logging.info(f"No results found for {filename}")

def main():
    """Parse command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze author name matches and differences.')
    parser.add_argument('--input_file', default='author_matches.json', help='Path to the input JSON file containing author matches')
    parser.add_argument('--output-dir', default='output',
                      help='Directory where output files will be saved (default: output)')
    
    args = parser.parse_args()
    analyze_author_matches(args.input_file, args.output_dir)

if __name__ == '__main__':
    main() 