"""
VLLM-based classifier for author name mismatches in academic citations.

This script uses VLLM to run inference with language models (default: Qwen/Qwen3-4B-Instruct-2507)
to classify author name discrepancies between citations and DBLP records.
"""

import json
import os
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch

# HuggingFace authentication
from huggingface_hub import login

# VLLM imports
from vllm import LLM, SamplingParams

# Local imports
from .prompt import create_classification_prompt

def setup_logging(output_dir: str) -> None:
    """Configure logging to write to both file and console."""
    log_file = os.path.join(output_dir, 'vllm_classifier.log')

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

def create_guided_json_schema() -> Dict[str, Any]:
    """
    Create the JSON schema for guided decoding.
    This ensures the model outputs valid JSON in the expected format.
    """
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "PARSER_ERROR", "NICKNAME", "MIDDLE_NAME", "INITIAL_VS_FULL",
                    "TRANSLITERATION", "DEADNAME", "NAME_CHANGE_OTHER",
                    "WRONG_PERSON", "TYPO", "AUTHOR_ORDER_ERROR", "LAST_NAME_ERROR",
                    "AUTHOR_MISSING", "AMBIGUOUS"
                ]
            },
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"]
            },
            "reasoning": {
                "type": "string",
                "maxLength": 100
            },
            "harm_level": {
                "type": "string",
                "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
            }
        },
        "required": ["category", "confidence", "reasoning", "harm_level"]
    }

def extract_result_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract classification result from LLM response.
    Expected format: Reasoning: [explanation] RESULT: [CATEGORY]
    Also handles variations like [reasoning] RESULT: [CATEGORY] or Category: [CATEGORY]
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Dict with category, reasoning, confidence, and harm_level, or None if parsing fails
    """
    import re
    
    # Try multiple patterns to extract category
    # Pattern 1: RESULT: [CATEGORY] (preferred format)
    result_patterns = [
        r'RESULT:\s*([A-Z_]+)',  # RESULT: TYPO
        r'\[([A-Z_]+)\]',  # [TYPO] at start or end
        r'Category:\s*([A-Z_]+)',  # Category: TYPO
        r'category:\s*([A-Z_]+)',  # category: TYPO
        r'RESULT\s+([A-Z_]+)',  # RESULT TYPO (no colon)
    ]
    
    category = None
    for pattern in result_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            category = match.group(1).upper()
            # Validate it's a known category
            valid_categories = [
                'PARSER_ERROR', 'NICKNAME', 'MIDDLE_NAME', 'INITIAL_VS_FULL',
                'TRANSLITERATION', 'DEADNAME', 'NAME_CHANGE_OTHER',
                'WRONG_PERSON', 'TYPO', 'AUTHOR_ORDER_ERROR', 'LAST_NAME_ERROR',
                'AUTHOR_MISSING', 'AMBIGUOUS'
            ]
            if category in valid_categories:
                break
            category = None
    
    # If no explicit category found, try to infer from reasoning text
    if not category:
        # Look for category names mentioned in the reasoning
        reasoning_lower = response_text.lower()
        category_keywords = {
            'typo': 'TYPO',
            'spelling error': 'TYPO',
            'nickname': 'NICKNAME',
            'deadname': 'DEADNAME',
            'parser error': 'PARSER_ERROR',
            'parsing error': 'PARSER_ERROR',
            'author missing': 'AUTHOR_MISSING',
            'author not found': 'AUTHOR_MISSING',
            'wrong person': 'WRONG_PERSON',
            'different person': 'WRONG_PERSON',
            'middle name': 'MIDDLE_NAME',
            'initial': 'INITIAL_VS_FULL',
            'transliteration': 'TRANSLITERATION',
            'western name': 'NICKNAME',  # Western names are now classified as NICKNAME
            'author order': 'AUTHOR_ORDER_ERROR',
            'last name error': 'LAST_NAME_ERROR',
            'last name mismatch': 'LAST_NAME_ERROR',
        }
        
        for keyword, cat in category_keywords.items():
            if keyword in reasoning_lower:
                category = cat
                break
    
    if not category:
        return None
    
    # Extract reasoning - look for meaningful reasoning text
    lines = response_text.split('\n')
    reasoning_lines = []
    
    # Skip placeholder lines and result lines
    skip_patterns = ['[CATEGORY]', 'RESULT:', 'Category:', 'category:', '[Your reasoning', 'Your reasoning', '[brief explanation]', 'brief explanation', 'Format:', 'Example:', 'Remember: DBLP']
    
    # First, try to find "Reasoning:" line (new format)
    for line in lines:
        line = line.strip()
        if line.startswith('Reasoning:') or line.startswith('reasoning:'):
            # Extract text after "Reasoning:"
            reasoning_text = line.split(':', 1)[1].strip() if ':' in line else line
            if reasoning_text and len(reasoning_text) > 10:
                reasoning_lines.append(reasoning_text)
                break
    
    # If no "Reasoning:" line found, look for other meaningful reasoning text
    if not reasoning_lines:
        for line in lines:
            line = line.strip()
            # Skip empty lines, result lines, and placeholders
            if not line or any(pattern.lower() in line.lower() for pattern in skip_patterns):
                continue
            # Skip lines that are just placeholders or too short
            if len(line) < 15:
                continue
            # Skip lines that look like format instructions
            if line.startswith('Format:') or line.startswith('Example:'):
                continue
            # Take first substantial reasoning line
            reasoning_lines.append(line)
            if len(reasoning_lines) >= 1:  # Just take one good line
                break
    
    # If still no reasoning, try to get any meaningful line
    if not reasoning_lines:
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not any(pattern in line for pattern in skip_patterns):
                reasoning_lines = [line]
                break
    
    reasoning = reasoning_lines[0] if reasoning_lines else "Classification based on mismatch analysis"
    
    # Determine confidence and harm_level based on category
    # Map categories to harm levels
    harm_level_map = {
        'DEADNAME': 'CRITICAL',
        'WRONG_PERSON': 'HIGH',
        'NICKNAME': 'LOW',
        'TYPO': 'LOW',
        'PARSER_ERROR': 'LOW',
        'INITIAL_VS_FULL': 'NONE',
        'MIDDLE_NAME': 'NONE',
        'AUTHOR_ORDER_ERROR': 'LOW',
        'LAST_NAME_ERROR': 'LOW',
        'AUTHOR_MISSING': 'MEDIUM',
        'TRANSLITERATION': 'LOW',
        'NAME_CHANGE_OTHER': 'MEDIUM',
        'AMBIGUOUS': 'LOW',
    }
    
    # Confidence based on category certainty
    high_confidence_categories = ['TYPO', 'AUTHOR_ORDER_ERROR', 'AUTHOR_MISSING', 'WRONG_PERSON']
    medium_confidence_categories = ['NICKNAME', 'INITIAL_VS_FULL', 'LAST_NAME_ERROR', 'PARSER_ERROR']
    
    if category in high_confidence_categories:
        confidence = 'HIGH'
    elif category in medium_confidence_categories:
        confidence = 'MEDIUM'
    else:
        confidence = 'MEDIUM'  # Default
    
    harm_level = harm_level_map.get(category, 'LOW')
    
    return {
        "category": category,
        "confidence": confidence,
        "reasoning": reasoning[:100],  # Limit length
        "harm_level": harm_level
    }

def classify_mismatch_batch(
    llm: LLM,
    records: List[Dict[str, Any]],
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Classify a batch of author name mismatches using VLLM.

    Optimized for speed with larger batch sizes and efficient JSON parsing.

    Args:
        llm: VLLM LLM instance
        records: List of validation records (with all original fields)
        batch_size: Number of samples to process in each batch

    Returns:
        List of records with llm_classification added
    """
    results = []

    # Prepare mismatch data for prompts (include all relevant fields)
    prompt_data_list = []
    for record in records:
        mismatch_data = {
            "reference": record.get("reference", {}),
            "dblp_match": record.get("dblp_match", {}),
            "mismatches": record.get("mismatches", []),
            "error_classifications": record.get("error_classifications", [])
        }
        prompt_data_list.append(mismatch_data)

    # Single progress bar for batches (less noisy than per-record updates)
    total_batches = (len(records) + batch_size - 1) // batch_size  # Ceiling division
    with tqdm(total=total_batches, desc=f"Processing batches (size={batch_size})") as pbar:
        # Process in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(records))

            batch_records = records[start_idx:end_idx]
            batch_prompt_data = prompt_data_list[start_idx:end_idx]

            # Create prompts for the batch
            prompts = [create_classification_prompt(data) for data in batch_prompt_data]

            # Optimized sampling parameters following Qwen3-4B-Instruct best practices
            # Recommended: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
            sampling_params = SamplingParams(
                temperature=0.7,  # Recommended for Qwen3 instruct models
                top_p=0.8,         # Recommended TopP
                top_k=20,          # Recommended TopK
                min_p=0.0,         # Recommended MinP
                max_tokens=500,    # Adequate for simple classification output
                presence_penalty=0.0,  # Can adjust 0-2 to reduce repetitions if needed
                stop=None,
            )

            try:
                # Generate responses (VLLM handles batching efficiently)
                # Temporarily suppress stdout to hide VLLM progress bars
                import sys
                from contextlib import redirect_stdout, redirect_stderr
                with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                    outputs = llm.generate(prompts, sampling_params)

                # Process each output in the batch
                for j, output in enumerate(outputs):
                    response_text = output.outputs[0].text.strip()
                    classification = extract_result_from_response(response_text)

                    # Create result with all original fields + classification
                    result = {
                        **batch_records[j],  # Preserve all original fields
                        "llm_classification": classification if classification else {
                            "error": "Result parsing failed - expected format: [reasoning] RESULT: [CATEGORY]",
                            "raw_response": response_text[:500]  # Limit error message size
                        }
                    }
                    results.append(result)

                    if not classification:
                        logging.warning(f"Failed to parse result for record {start_idx+j}: {response_text[:100]}")

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                # Add error results for this batch
                for j, record in enumerate(batch_records):
                    results.append({
                        **record,
                        "llm_classification": {
                            "error": f"Batch processing failed: {str(e)}"
                        }
                    })

            pbar.update(1)  # Update progress for each batch completed

    return results

def load_validation_data(input_file: str, max_records: int = None) -> List[Dict[str, Any]]:
    """
    Load validation results from JSON file, preserving all original fields.
    
    Returns list of records with all original fields preserved, ready for classification.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        records = []
        count = 0

        # Extract records from nested structure
        def extract_records(obj):
            nonlocal count, records
            if isinstance(obj, dict):
                if 'results' in obj and isinstance(obj['results'], list):
                    # Found a results array
                    for record in obj['results']:
                        if max_records and count >= max_records:
                            return
                        if isinstance(record, dict) and 'reference' in record:
                            records.append(record)  # Keep full record
                            count += 1
                elif 'mismatches' in obj and 'reference' in obj:
                    # Direct mismatch record
                    if max_records and count >= max_records:
                        return
                    records.append(obj)
                    count += 1
                else:
                    # Recursively search in nested dicts
                    for value in obj.values():
                        extract_records(value)
            elif isinstance(obj, list):
                # Recursively search in lists
                for item in obj:
                    extract_records(item)

        extract_records(raw_data)

        logging.info(f"Loaded {len(records)} validation records from {input_file}")
        return records

    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in '{input_file}'.")
        raise

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save classification results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(results)} classified results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}")
        raise

def main():
    """Main function to run VLLM classification."""
    parser = argparse.ArgumentParser(
        description='Classify author name mismatches using VLLM and Qwen/Qwen3-4B-Instruct-2507'
    )
    parser.add_argument(
        '--input_file',
        default='../validation_results/validation_results.json',
        help='Path to the validation results JSON file'
    )
    parser.add_argument(
        '--output_file',
        default='../validation_results/classified_results.json',
        help='Path where classified results will be saved'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of samples to process in each batch (larger = faster but more memory)'
    )
    parser.add_argument(
        '--model_name',
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='HuggingFace model name to use'
    )
    parser.add_argument(
        '--hf_token',
        default=os.getenv('HF_TOKEN', None),
        help='HuggingFace authentication token (or set HF_TOKEN environment variable)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization (0-1)'
    )

    args = parser.parse_args()

    # Validate HuggingFace token
    if not args.hf_token:
        raise ValueError(
            "HuggingFace token is required. Set HF_TOKEN environment variable or use --hf_token argument.\n"
            "Example: export HF_TOKEN=your_token_here"
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    logging.info("Starting VLLM classification pipeline")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Input: {args.input_file}")
    logging.info(f"Output: {args.output_file}")
    logging.info(f"Batch size: {args.batch_size}")

    try:
        # Set environment variables to reduce VLLM verbosity
        os.environ['VLLM_LOGGING_LEVEL'] = 'CRITICAL'  # Reduce VLLM's logging to critical only
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer warnings

        # Authenticate with HuggingFace
        logging.info("Authenticating with HuggingFace...")
        login(token=args.hf_token)

        # Load validation data
        max_records = args.max_samples if args.max_samples else None
        validation_data = load_validation_data(args.input_file, max_records=max_records)
        logging.info(f"Processing {len(validation_data)} samples")

        # Initialize VLLM (optimized settings)
        # For Qwen3-4B-Thinking models, reasoning is automatically detected
        logging.info("Initializing VLLM...")
        llm_kwargs = {
            "model": args.model_name,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": torch.bfloat16,  # Use bfloat16 for efficient inference
            "max_model_len": 4096,    # Increased to handle longer prompts with detailed categories
            "tensor_parallel_size": 1,  # Adjust if you have multiple GPUs
            "trust_remote_code": True,
            "max_num_seqs": 32,  # Limit concurrent sequences to reduce memory
        }
        
        # Note: For thinking models, reasoning would be auto-detected
        # For instruct models, no special configuration needed
        
        llm = LLM(**llm_kwargs)

        # Run classification
        logging.info("Starting classification...")
        results = classify_mismatch_batch(llm, validation_data, args.batch_size)

        # Save results
        save_results(results, args.output_file)

        # Log summary statistics
        successful_classifications = sum(
            1 for r in results
            if 'llm_classification' in r and 'error' not in r['llm_classification']
        )
        error_count = len(results) - successful_classifications

        logging.info(f"Classification complete!")
        logging.info(f"Total processed: {len(results)}")
        logging.info(f"Successful classifications: {successful_classifications}")
        logging.info(f"Errors: {error_count}")

        # Show sample of results
        if results:
            logging.info("Sample classification result:")
            sample = results[0]
            if 'llm_classification' in sample and 'error' not in sample['llm_classification']:
                logging.info(json.dumps(sample['llm_classification'], indent=2))

    except Exception as e:
        logging.error(f"Fatal error during classification: {e}")
        raise

if __name__ == '__main__':
    main()
