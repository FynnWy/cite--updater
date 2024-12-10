import os
import random
import argparse
from pathlib import Path
from citation_pipeline import process_publications
from tqdm import tqdm
from time import sleep
def sample_xml_files(data_dir: str, n: int) -> list:
    """Sample n random XML files from the data directory."""
    xml_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    if not xml_files:
        raise ValueError(f"No XML files found in {data_dir}")
    
    n = min(n, len(xml_files))  # Ensure we don't try to sample more than available
    return random.sample(xml_files, n)

def main():
    parser = argparse.ArgumentParser(description='Sample and process XML files using citation pipeline')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing XML files')
    parser.add_argument('--n-samples', type=int, default=15500,
                       help='Number of XML files to sample')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to store output JSON files')
    parser.add_argument('--threshold', type=int, default=90,
                       help='Title match threshold (0-100)')
    parser.add_argument('--dblp-delay', type=float, default=1.0,
                       help='Delay between DBLP API calls in seconds')
    parser.add_argument('--arxiv-delay', type=float, default=1.0,
                       help='Delay between arXiv API calls in seconds')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sample XML files
    print(f"Sampling {args.n_samples} XML files from {args.data_dir}")
    sampled_files = sample_xml_files(args.data_dir, args.n_samples)
    
    # Process each sampled file with progress bar
    for input_file in tqdm(sampled_files, desc="Processing files", unit="file"):
        # Generate output filename
        input_filename = Path(input_file).stem
        output_file = os.path.join(args.output_dir, f"{input_filename}_matches.json")
        
        # Process the file
        process_publications(
            input_file=input_file,
            output_file=output_file,
            match_threshold=args.threshold,
            dblp_delay=args.dblp_delay,
            arxiv_delay=args.arxiv_delay,
            dry_run=False
        )

if __name__ == '__main__':
    main() 