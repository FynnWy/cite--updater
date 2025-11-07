"""
Parse citations from GROBID XML files and combine them into a single JSON file.

This script processes all XML files in the sample folder and extracts citation information
including paper title, authors, proceedings/journal, and year.

Usage:
    python src/parse_citations.py --input sample/ --output citations.json
"""

import xml.etree.ElementTree as ET
import json
import os
import argparse
from typing import List, Dict, Optional
from pathlib import Path


def extract_text(element: Optional[ET.Element]) -> Optional[str]:
    """Extract text content from an XML element, handling None cases."""
    if element is not None and element.text:
        return element.text.strip()
    return None


def extract_all_text(element: Optional[ET.Element]) -> str:
    """Extract all text content from an element and its children."""
    if element is None:
        return ""
    # Use itertext() to get all text content recursively
    return ''.join(element.itertext()).strip()


def parse_citations_from_xml(file_path: str) -> List[Dict[str, any]]:
    """
    Parse citations from a single GROBID XML file.
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        List of citation dictionaries, each containing title, authors, proceedings, and year
    """
    citations = []
    
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Define the namespace used by GROBID TEI XML
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Find all biblStruct elements (these are the citations)
        # They can be in listBibl (references section) or elsewhere
        for biblStruct in root.findall('.//tei:biblStruct', ns):
            citation = {
                'title': None,
                'authors': [],
                'proceedings': None,
                'year': None
            }
            
            # Extract title from analytic section (paper title)
            analytic = biblStruct.find('.//tei:analytic', ns)
            if analytic is not None:
                title_elem = analytic.find('.//tei:title[@type="main"]', ns)
                if title_elem is None:
                    # Try without type attribute
                    title_elem = analytic.find('.//tei:title[@level="a"]', ns)
                if title_elem is not None:
                    citation['title'] = extract_all_text(title_elem)
                
                # Extract authors from analytic section
                for author in analytic.findall('.//tei:author', ns):
                    pers_name = author.find('.//tei:persName', ns)
                    if pers_name is not None:
                        # Extract forename(s) and surname
                        forenames = []
                        for forename in pers_name.findall('tei:forename', ns):
                            fn_text = extract_text(forename)
                            if fn_text:
                                forenames.append(fn_text)
                        
                        surname_elem = pers_name.find('tei:surname', ns)
                        surname = extract_text(surname_elem)
                        
                        # Construct full name
                        if forenames and surname:
                            full_name = ' '.join(forenames) + ' ' + surname
                            citation['authors'].append(full_name)
                        elif surname:
                            citation['authors'].append(surname)
            
            # If no title in analytic, check monogr section
            if not citation['title']:
                monogr = biblStruct.find('.//tei:monogr', ns)
                if monogr is not None:
                    # Check for title in monogr (could be book/journal title or paper title)
                    title_elem = monogr.find('.//tei:title[@type="main"]', ns)
                    if title_elem is None:
                        title_elem = monogr.find('.//tei:title[@level="m"]', ns)
                    if title_elem is not None:
                        citation['title'] = extract_all_text(title_elem)
            
            # Extract proceedings/journal/conference name from monogr section
            monogr = biblStruct.find('.//tei:monogr', ns)
            if monogr is not None:
                # Try meeting element first (conference name)
                meeting_elem = monogr.find('tei:meeting', ns)
                if meeting_elem is not None:
                    # Extract text from meeting, but exclude address if present
                    meeting_text = extract_text(meeting_elem)
                    if meeting_text:
                        citation['proceedings'] = meeting_text
                
                # If no meeting, try journal title (level="j") or proceedings title (level="m")
                if not citation['proceedings']:
                    journal_title = monogr.find('.//tei:title[@level="j"]', ns)
                    if journal_title is not None:
                        citation['proceedings'] = extract_all_text(journal_title)
                    else:
                        # Check for proceedings title (level="m") but only if it's not the paper title
                        proc_title = monogr.find('.//tei:title[@level="m"]', ns)
                        if proc_title is not None:
                            proc_text = extract_all_text(proc_title)
                            # Only use if it's different from the paper title
                            if proc_text and proc_text != citation['title']:
                                citation['proceedings'] = proc_text
            
            # Extract year from imprint/date
            imprint = biblStruct.find('.//tei:imprint', ns)
            if imprint is not None:
                date_elem = imprint.find('.//tei:date[@type="published"]', ns)
                if date_elem is None:
                    # Try any date element
                    date_elem = imprint.find('.//tei:date', ns)
                
                if date_elem is not None:
                    # Try to get year from 'when' attribute first
                    year_attr = date_elem.get('when')
                    if year_attr:
                        # Extract year (first 4 digits)
                        try:
                            citation['year'] = int(year_attr[:4])
                        except (ValueError, TypeError):
                            pass
                    
                    # If no 'when' attribute, try text content
                    if citation['year'] is None:
                        date_text = extract_text(date_elem)
                        if date_text:
                            # Try to extract year from text (look for 4-digit number)
                            import re
                            year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
                            if year_match:
                                try:
                                    citation['year'] = int(year_match.group())
                                except (ValueError, TypeError):
                                    pass
            
            # Only add citation if it has at least a title or authors
            if citation['title'] or citation['authors']:
                citations.append(citation)
    
    except ET.ParseError as e:
        print(f"XML parsing error in {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return citations


def parse_all_xml_files(input_dir: str) -> List[Dict[str, any]]:
    """
    Parse all XML files in the input directory and collect all citations.
    
    Args:
        input_dir: Directory containing XML files
        
    Returns:
        List of all citations from all XML files
    """
    all_citations = []
    xml_files = list(Path(input_dir).glob('*.xml'))
    
    print(f"Found {len(xml_files)} XML files to process...")
    
    for i, xml_file in enumerate(xml_files, 1):
        print(f"Processing {i}/{len(xml_files)}: {xml_file.name}")
        citations = parse_citations_from_xml(str(xml_file))
        print(f"  Found {len(citations)} citations")
        all_citations.extend(citations)
    
    print(f"\nTotal citations extracted: {len(all_citations)}")
    return all_citations


def main():
    """Main function to parse citations from XML files and save to JSON."""
    parser = argparse.ArgumentParser(
        description='Parse citations from GROBID XML files and combine into JSON'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='sample/',
        help='Input directory containing XML files (default: sample/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='citations.json',
        help='Output JSON file path (default: citations.json)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Parse all XML files
    all_citations = parse_all_xml_files(args.input)
    
    # Save to JSON file
    print(f"\nSaving {len(all_citations)} citations to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_citations, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Citations saved to {args.output}")


if __name__ == '__main__':
    main()

