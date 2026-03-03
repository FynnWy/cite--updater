"""
Script to parse GROBID TEI XML files and extract metadata to CSV format.

This script processes all files matching the pattern '2025.*.grobid.tei.xml' 
in the data/outputs/arxiv_pdfs directory, extracts ID, Title, Authors, and 
Affiliations, and writes them to a CSV file.
"""

import argparse
import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

# Define XML namespace used by GROBID TEI files
TEI_NAMESPACE = {'tei': 'http://www.tei-c.org/ns/1.0'}


def extract_id_from_filename(filename: str) -> str:
    """
    Extract ID from filename by removing the .grobid.tei.xml extension.
    
    Args:
        filename: The filename (e.g., '2025.acl-long.1550.grobid.tei.xml')
        
    Returns:
        The ID (e.g., '2025.acl-long.1550')
    """
    # Remove the .grobid.tei.xml extension
    if filename.endswith('.grobid.tei.xml'):
        return filename[:-15]  # Remove '.grobid.tei.xml' (15 characters)
    return filename


def extract_title(root: ET.Element) -> Optional[str]:
    """
    Extract the main title from the TEI XML structure.
    
    Looks for title elements in the titleStmt or analytic sections.
    
    Args:
        root: The root element of the parsed XML tree
        
    Returns:
        The title string, or None if not found
    """
    # Try to find title in titleStmt first
    title_stmt = root.find('.//tei:titleStmt', TEI_NAMESPACE)
    if title_stmt is not None:
        title_elem = title_stmt.find('tei:title[@level="a"][@type="main"]', TEI_NAMESPACE)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
    
    # Fallback: try to find title in analytic section
    analytic = root.find('.//tei:analytic', TEI_NAMESPACE)
    if analytic is not None:
        title_elem = analytic.find('tei:title[@level="a"][@type="main"]', TEI_NAMESPACE)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
    
    return None


def extract_authors(root: ET.Element) -> str:
    """
    Extract all authors from the TEI XML structure.
    
    Only extracts authors from the analytic section (main paper authors),
    not from citations/references. Combines forename and surname for each 
    author, separated by semicolons.
    
    Args:
        root: The root element of the parsed XML tree
        
    Returns:
        Semicolon-separated string of author names (e.g., 'John Doe; Jane Smith')
    """
    authors = []
    
    # Find the analytic section within biblStruct in sourceDesc
    # This restricts to only the main paper authors, not citations
    analytic = root.find('.//tei:sourceDesc/tei:biblStruct/tei:analytic', TEI_NAMESPACE)
    
    if analytic is not None:
        # Find all author elements only within the analytic section
        author_elems = analytic.findall('tei:author', TEI_NAMESPACE)
        
        for author_elem in author_elems:
            # Extract persName element
            pers_name = author_elem.find('tei:persName', TEI_NAMESPACE)
            if pers_name is not None:
                # Extract forename and surname
                forename_elem = pers_name.find('tei:forename', TEI_NAMESPACE)
                surname_elem = pers_name.find('tei:surname', TEI_NAMESPACE)
                
                forename = forename_elem.text.strip() if forename_elem is not None and forename_elem.text else ''
                surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ''
                
                # Combine forename and surname
                if forename or surname:
                    author_name = f"{forename} {surname}".strip()
                    if author_name:
                        authors.append(author_name)
    
    return '; '.join(authors) if authors else ''


def extract_affiliations(root: ET.Element) -> str:
    """
    Extract all affiliations from the TEI XML structure.
    
    Only extracts affiliations from the analytic section (main paper authors' affiliations),
    not from citations/references. Extracts organization names from affiliation elements, 
    separated by semicolons.
    
    Args:
        root: The root element of the parsed XML tree
        
    Returns:
        Semicolon-separated string of affiliations
    """
    affiliations = []
    
    # Find the analytic section within biblStruct in sourceDesc
    # This restricts to only the main paper authors' affiliations, not citations
    analytic = root.find('.//tei:sourceDesc/tei:biblStruct/tei:analytic', TEI_NAMESPACE)
    
    if analytic is not None:
        # Find all affiliation elements only within the analytic section
        affiliation_elems = analytic.findall('.//tei:affiliation', TEI_NAMESPACE)
        
        for affil_elem in affiliation_elems:
            # Look for orgName elements within the affiliation
            org_names = affil_elem.findall('.//tei:orgName', TEI_NAMESPACE)
            
            for org_name in org_names:
                if org_name.text:
                    affiliation_text = org_name.text.strip()
                    if affiliation_text and affiliation_text not in affiliations:
                        affiliations.append(affiliation_text)
            
            # If no orgName found, try to get text from addrLine or other elements
            if not org_names:
                addr_line = affil_elem.find('.//tei:addrLine', TEI_NAMESPACE)
                if addr_line is not None and addr_line.text:
                    affiliation_text = addr_line.text.strip()
                    if affiliation_text and affiliation_text not in affiliations:
                        affiliations.append(affiliation_text)
    
    return '; '.join(affiliations) if affiliations else ''


def parse_xml_file(file_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse a single GROBID TEI XML file and extract metadata.
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        Dictionary with keys: 'ID', 'Title', 'Authors', 'Affiliations'
        Returns None if parsing fails
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract ID from filename
        file_id = extract_id_from_filename(file_path.name)
        
        # Extract metadata
        title = extract_title(root)
        authors = extract_authors(root)
        affiliations = extract_affiliations(root)
        
        return {
            'ID': file_id,
            'Title': title or '',
            'Authors': authors,
            'Affiliations': affiliations
        }
    
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return None


def process_all(input_dir: Path, pattern: str, output_csv: Path) -> None:
    """Process all matching XML files and write a TSV."""
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    xml_files = list(input_dir.glob(pattern))
    if not xml_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to process...")
    
    results: List[Dict[str, str]] = []
    for xml_file in sorted(xml_files):
        result = parse_xml_file(xml_file)
        if result:
            results.append(result)
    
    if results:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ID', 'Title', 'Authors', 'Affiliations']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"Successfully processed {len(results)} files.")
        print(f"CSV file written to: {output_csv}")
    else:
        print("No valid data extracted from XML files.")


def main():
    parser = argparse.ArgumentParser(description="Parse GROBID TEI XML into TSV/CSV")
    parser.add_argument('--input-dir', type=Path, default=Path('data/outputs/arxiv_pdfs'))
    parser.add_argument('--pattern', type=str, default='*.grobid.tei.xml')
    parser.add_argument('--output-csv', type=Path, default=Path('data/arxiv_metadata.csv'))
    args = parser.parse_args()
    process_all(args.input_dir, args.pattern, args.output_csv)


if __name__ == '__main__':
    main()

