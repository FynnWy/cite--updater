"""
Process validation results to add LLM classifications.

This script processes the large validation_results.json file and adds LLM classifications
to create an enhanced dataset suitable for prompt loop training/validation.
"""

import json
import os
import logging
import argparse
from typing import Dict, Any, Iterator
from tqdm import tqdm

def setup_logging(output_dir: str) -> None:
    """Configure logging."""
    log_file = os.path.join(output_dir, 'process_validation.log')

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def stream_json_array(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream JSON objects from a large JSON array file.

    Args:
        file_path: Path to JSON file containing an array of objects

    Yields:
        Individual JSON objects from the array
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip whitespace and find the opening bracket
            while True:
                char = f.read(1)
                if char == '[':
                    break
                elif char == '':
                    raise ValueError("File does not contain a JSON array")

            # Read objects one by one
            brace_count = 0
            current_object = ""
            in_string = False
            escape_next = False

            while True:
                char = f.read(1)
                if char == '':
                    break

                current_object += char

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                        # When brace count returns to 0, we have a complete object
                        if brace_count == 0:
                            # Remove the comma if present
                            obj_str = current_object.rstrip(',')
                            try:
                                obj = json.loads(obj_str)
                                yield obj
                                current_object = ""
                            except json.JSONDecodeError:
                                # Skip malformed objects
                                logging.warning(f"Skipping malformed JSON object")
                                current_object = ""

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error streaming JSON from {file_path}: {e}")
        raise

def create_enhanced_validation_record(
    original_record: Dict[str, Any],
    llm_classification: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create an enhanced validation record with LLM classification.

    Args:
        original_record: Original validation record
        llm_classification: LLM classification result (optional)

    Returns:
        Enhanced record with classification info
    """
    enhanced = {
        # Original mismatch data for prompt input
        "mismatch_data": {
            "reference": original_record.get("reference", {}),
            "dblp_match": original_record.get("dblp_match", {}),
            "mismatches": original_record.get("mismatches", []),
            "error_classifications": original_record.get("error_classifications", []),
            "source_paper": original_record.get("source_paper", {}),
            "validation_status": original_record.get("validation_status", "")
        },

        # Ground truth information
        "ground_truth": {
            "has_author_mismatch": "author_mismatch" in original_record.get("validation_status", ""),
            "mismatch_types": original_record.get("error_classifications", []),
            "severity_indicators": []  # Could be populated based on domain knowledge
        },

        # LLM classification (to be filled by VLLM classifier)
        "llm_classification": llm_classification or {},

        # Metadata
        "metadata": {
            "processed_timestamp": None,  # To be set when processing
            "model_used": "google/gemma-3-4b-it",
            "prompt_version": "1.0"
        }
    }

    return enhanced

def create_sample_dataset(
    input_file: str,
    output_file: str,
    sample_size: int = 1000,
    categories: list = None
) -> None:
    """
    Create a smaller sample dataset for testing and development.

    Args:
        input_file: Path to full validation_results.json
        output_file: Path to save sample dataset
        sample_size: Number of samples to include
        categories: Specific categories to sample from (optional)
    """
    logging.info(f"Creating sample dataset with {sample_size} records")

    samples_per_category = {}
    if categories:
        samples_per_category = {cat: sample_size // len(categories) for cat in categories}

    collected_samples = []
    category_counts = {}

    for record in stream_json_array(input_file):
        error_types = record.get("error_classifications", [])

        # Determine category for stratified sampling
        category = "other"
        if "first_name_mismatch" in error_types:
            category = "first_name"
        elif "last_name_mismatch" in error_types:
            category = "last_name"
        elif "parsing_errors" in str(error_types).lower():
            category = "parsing"

        # Check if we should include this record
        should_include = False
        if categories and category in categories:
            if category_counts.get(category, 0) < samples_per_category[category]:
                should_include = True
                category_counts[category] = category_counts.get(category, 0) + 1
        elif not categories and len(collected_samples) < sample_size:
            should_include = True

        if should_include:
            enhanced_record = create_enhanced_validation_record(record)
            collected_samples.append(enhanced_record)

        # Break if we have enough samples
        if categories:
            total_collected = sum(category_counts.values())
            if total_collected >= sum(samples_per_category.values()):
                break
        elif len(collected_samples) >= sample_size:
            break

    # Save sample dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(collected_samples, f, indent=2, ensure_ascii=False)

    logging.info(f"Created sample dataset with {len(collected_samples)} records")
    if categories:
        logging.info(f"Category breakdown: {category_counts}")

def process_full_dataset(input_file: str, output_file: str) -> None:
    """
    Process the full validation results dataset.

    Args:
        input_file: Path to full validation_results.json
        output_file: Path to save enhanced dataset
    """
    logging.info("Processing full validation dataset")

    record_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # Start JSON array

        first_record = True
        for record in stream_json_array(input_file):
            enhanced_record = create_enhanced_validation_record(record)
            record_count += 1

            # Write record to file
            if not first_record:
                f.write(',\n')
            json.dump(enhanced_record, f, indent=2, ensure_ascii=False)
            first_record = False

            if record_count % 1000 == 0:
                logging.info(f"Processed {record_count} records")

        f.write('\n]')  # End JSON array

    logging.info(f"Processed {record_count} total records")

def merge_classifications(
    original_file: str,
    classifications_file: str,
    output_file: str
) -> None:
    """
    Merge LLM classifications back into the enhanced validation results.

    Args:
        original_file: Path to enhanced validation results (without classifications)
        classifications_file: Path to file with LLM classifications
        output_file: Path to save merged results
    """
    logging.info("Merging LLM classifications")

    # Load classifications
    with open(classifications_file, 'r', encoding='utf-8') as f:
        classifications = json.load(f)

    # Create lookup dict by some unique identifier
    # For now, we'll use the reference ID and title as composite key
    classification_lookup = {}
    for item in classifications:
        ref = item.get("reference", {})
        key = f"{ref.get('id', '')}_{ref.get('title', '')}"
        classification_lookup[key] = item.get("llm_classification", {})

    logging.info(f"Loaded {len(classification_lookup)} classifications")

    # Process original file and merge
    merged_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        first_record = True

        for record in stream_json_array(original_file):
            # Create lookup key
            mismatch_data = record.get("mismatch_data", {})
            ref = mismatch_data.get("reference", {})
            key = f"{ref.get('id', '')}_{ref.get('title', '')}"

            # Get classification if available
            classification = classification_lookup.get(key, {})

            # Update record with classification
            record["llm_classification"] = classification

            # Write merged record
            if not first_record:
                f.write(',\n')
            json.dump(record, f, indent=2, ensure_ascii=False)
            first_record = False

            merged_count += 1
            if merged_count % 1000 == 0:
                logging.info(f"Merged {merged_count} records")

        f.write('\n]')

    logging.info(f"Merged classifications for {merged_count} records")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process validation results for LLM classification'
    )
    parser.add_argument(
        'action',
        choices=['sample', 'process_full', 'merge'],
        help='Action to perform'
    )
    parser.add_argument(
        '--input_file',
        default='validation_results/validation_results.json',
        help='Path to input validation results'
    )
    parser.add_argument(
        '--output_file',
        default='validation_results/enhanced_validation_results.json',
        help='Path for output file'
    )
    parser.add_argument(
        '--classifications_file',
        help='Path to classifications file (for merge action)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=1000,
        help='Sample size for sample action'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        help='Categories to sample from'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    if args.action == 'sample':
        create_sample_dataset(
            args.input_file,
            args.output_file,
            args.sample_size,
            args.categories
        )
    elif args.action == 'process_full':
        process_full_dataset(args.input_file, args.output_file)
    elif args.action == 'merge':
        if not args.classifications_file:
            parser.error("--classifications_file required for merge action")
        merge_classifications(
            args.input_file,
            args.classifications_file,
            args.output_file
        )

if __name__ == '__main__':
    main()
