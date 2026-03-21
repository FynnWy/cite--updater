"""
LLM-based classifier for author name mismatches in academic citations.

Primary backend: vLLM (Linux/GPU).
Fallback backend: transformers (macOS/Windows/Linux CPU/MPS/CUDA).
"""

import json
import os
import logging
import argparse
import platform
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing optional dependency 'torch'. Install with "
        "'pip install -r requirements.txt -r requirements-llm.txt'."
    ) from exc

# HuggingFace authentication
from huggingface_hub import login

# Optional vLLM imports
VLLM_AVAILABLE = False
VLLM_IMPORT_ERROR = None
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as exc:
    VLLM_IMPORT_ERROR = exc
    LLM = Any  # type: ignore[assignment]
    SamplingParams = Any  # type: ignore[assignment]
else:
    VLLM_AVAILABLE = True

# Optional transformers imports
TRANSFORMERS_AVAILABLE = False
TRANSFORMERS_IMPORT_ERROR = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:
    TRANSFORMERS_IMPORT_ERROR = exc
    AutoModelForCausalLM = Any  # type: ignore[assignment]
    AutoTokenizer = Any  # type: ignore[assignment]
else:
    TRANSFORMERS_AVAILABLE = True

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

def prepare_prompt_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract the prompt-relevant subset from full validation records."""
    prompt_data_list = []
    for record in records:
        prompt_data_list.append(
            {
                "reference": record.get("reference", {}),
                "dblp_match": record.get("dblp_match", {}),
                "mismatches": record.get("mismatches", []),
                "error_classifications": record.get("error_classifications", []),
            }
        )
    return prompt_data_list


def classify_mismatch_batch_vllm(
    llm: Any,
    records: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Classify mismatch records with a vLLM backend."""
    results = []
    prompt_data_list = prepare_prompt_data(records)
    total_batches = (len(records) + batch_size - 1) // batch_size

    with tqdm(total=total_batches, desc=f"Processing batches (size={batch_size})") as pbar:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(records))

            batch_records = records[start_idx:end_idx]
            batch_prompt_data = prompt_data_list[start_idx:end_idx]
            prompts = [create_classification_prompt(data) for data in batch_prompt_data]

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                max_tokens=500,
                presence_penalty=0.0,
                stop=None,
            )

            try:
                outputs = llm.generate(prompts, sampling_params)

                for j, output in enumerate(outputs):
                    response_text = output.outputs[0].text.strip()
                    classification = extract_result_from_response(response_text)
                    results.append(
                        {
                            **batch_records[j],
                            "llm_classification": classification
                            if classification
                            else {
                                "error": "Result parsing failed - expected format: [reasoning] RESULT: [CATEGORY]",
                                "raw_response": response_text[:500],
                            },
                        }
                    )
                    if not classification:
                        logging.warning(
                            "Failed to parse result for record %s: %s",
                            start_idx + j,
                            response_text[:100],
                        )

            except Exception as exc:
                logging.error("Error processing batch %s: %s", batch_idx, exc)
                for record in batch_records:
                    results.append(
                        {
                            **record,
                            "llm_classification": {
                                "error": f"Batch processing failed: {str(exc)}"
                            },
                        }
                    )

            pbar.update(1)

    return results


def get_transformers_device(requested_device: str) -> str:
    """Resolve requested transformers device to an available runtime device."""
    if requested_device in {"cpu", "mps", "cuda"}:
        if requested_device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            logging.warning("Requested device 'mps' is unavailable. Falling back to CPU.")
            return "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            logging.warning("Requested device 'cuda' is unavailable. Falling back to CPU.")
            return "cpu"
        return requested_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_transformers_backend(model_name: str, requested_device: str) -> Tuple[Any, Any, str]:
    """Load tokenizer/model for the transformers fallback backend."""
    device = get_transformers_device(requested_device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    except ImportError as exc:
        err_text = str(exc)
        if "requires the PyTorch library" in err_text:
            raise RuntimeError(
                "Transformers cannot use the installed torch runtime. "
                "This is commonly caused by transformers>=5 with torch<2.4. "
                "Reinstall with: "
                "pip install --upgrade 'transformers<5' 'torch>=2.2.0,<2.4' "
                "and then rerun classification."
            ) from exc
        raise
    model.to(device)
    model.eval()

    logging.info("Transformers backend initialized on device: %s", device)
    return model, tokenizer, device


def classify_mismatch_batch_transformers(
    model: Any,
    tokenizer: Any,
    device: str,
    records: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Classify mismatch records with a transformers backend."""
    results = []
    prompt_data_list = prepare_prompt_data(records)
    total_batches = (len(records) + batch_size - 1) // batch_size

    model_max_len = getattr(tokenizer, "model_max_length", 4096)
    if not isinstance(model_max_len, int) or model_max_len <= 0 or model_max_len > 100000:
        model_max_len = 4096

    with tqdm(total=total_batches, desc=f"Processing batches (size={batch_size})") as pbar:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(records))

            batch_records = records[start_idx:end_idx]
            batch_prompt_data = prompt_data_list[start_idx:end_idx]
            prompts = [create_classification_prompt(data) for data in batch_prompt_data]

            try:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(model_max_len, 4096),
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_lengths = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        max_new_tokens=500,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for j in range(len(batch_records)):
                    response_ids = generated_ids[j][int(input_lengths[j]):].detach().cpu()
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                    classification = extract_result_from_response(response_text)
                    results.append(
                        {
                            **batch_records[j],
                            "llm_classification": classification
                            if classification
                            else {
                                "error": "Result parsing failed - expected format: [reasoning] RESULT: [CATEGORY]",
                                "raw_response": response_text[:500],
                            },
                        }
                    )
                    if not classification:
                        logging.warning(
                            "Failed to parse result for record %s: %s",
                            start_idx + j,
                            response_text[:100],
                        )

            except Exception as exc:
                logging.error("Error processing batch %s: %s", batch_idx, exc)
                for record in batch_records:
                    results.append(
                        {
                            **record,
                            "llm_classification": {
                                "error": f"Batch processing failed: {str(exc)}"
                            },
                        }
                    )

            pbar.update(1)

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

def resolve_backend(requested_backend: str) -> str:
    """Resolve backend choice based on availability and platform."""
    if requested_backend == "vllm":
        if not VLLM_AVAILABLE:
            raise ModuleNotFoundError(
                "Requested backend 'vllm' is unavailable. Install optional deps with "
                "'pip install -r requirements.txt -r requirements-llm.txt' and run on Linux."
            ) from VLLM_IMPORT_ERROR
        return "vllm"

    if requested_backend == "transformers":
        if not TRANSFORMERS_AVAILABLE:
            raise ModuleNotFoundError(
                "Requested backend 'transformers' is unavailable. Install optional deps with "
                "'pip install -r requirements.txt -r requirements-llm.txt'."
            ) from TRANSFORMERS_IMPORT_ERROR
        return "transformers"

    if platform.system() == "Linux" and VLLM_AVAILABLE:
        return "vllm"
    if TRANSFORMERS_AVAILABLE:
        return "transformers"
    if VLLM_AVAILABLE:
        return "vllm"

    raise ModuleNotFoundError(
        "No LLM backend available. Install optional deps with "
        "'pip install -r requirements.txt -r requirements-llm.txt'."
    )


def main():
    """Main function to run mismatch classification."""
    parser = argparse.ArgumentParser(
        description="Classify author name mismatches using vLLM or transformers."
    )
    parser.add_argument(
        "--input_file",
        default="../validation_results/validation_results.json",
        help="Path to the validation results JSON file",
    )
    parser.add_argument(
        "--output_file",
        default="../validation_results/classified_results.json",
        help="Path where classified results will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to process in each batch",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model name to use",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "transformers"],
        default="auto",
        help="Inference backend: auto (default), vllm, or transformers",
    )
    parser.add_argument(
        "--transformers_device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Runtime device for transformers backend",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN", None),
        help="HuggingFace token (optional for public models)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0-1)",
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Starting classification pipeline")
    logging.info("Model: %s", args.model_name)
    logging.info("Input: %s", args.input_file)
    logging.info("Output: %s", args.output_file)
    logging.info("Batch size: %s", args.batch_size)

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if args.hf_token:
            logging.info("Authenticating with HuggingFace...")
            login(token=args.hf_token)
        else:
            logging.info("HF_TOKEN not provided; continuing without authentication.")

        max_records = args.max_samples if args.max_samples else None
        validation_data = load_validation_data(args.input_file, max_records=max_records)
        logging.info("Processing %s samples", len(validation_data))

        backend = resolve_backend(args.backend)
        logging.info("Selected backend: %s", backend)

        if backend == "vllm":
            os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
            if platform.system() != "Linux":
                logging.warning("vLLM backend outside Linux may be unstable on this platform.")

            llm_kwargs = {
                "model": args.model_name,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "dtype": torch.bfloat16,
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "trust_remote_code": True,
                "max_num_seqs": 32,
            }
            logging.info("Initializing vLLM backend...")
            llm = LLM(**llm_kwargs)
            effective_batch_size = args.batch_size
            results = classify_mismatch_batch_vllm(llm, validation_data, effective_batch_size)
        else:
            logging.info("Initializing transformers backend...")
            model, tokenizer, device = init_transformers_backend(
                model_name=args.model_name,
                requested_device=args.transformers_device,
            )

            effective_batch_size = args.batch_size
            if device in {"cpu", "mps"} and effective_batch_size > 4:
                logging.info(
                    "Reducing batch_size from %s to 4 for transformers on %s.",
                    effective_batch_size,
                    device,
                )
                effective_batch_size = 4

            results = classify_mismatch_batch_transformers(
                model=model,
                tokenizer=tokenizer,
                device=device,
                records=validation_data,
                batch_size=effective_batch_size,
            )

        save_results(results, args.output_file)

        successful_classifications = sum(
            1
            for r in results
            if "llm_classification" in r and "error" not in r["llm_classification"]
        )
        error_count = len(results) - successful_classifications

        logging.info("Classification complete")
        logging.info("Total processed: %s", len(results))
        logging.info("Successful classifications: %s", successful_classifications)
        logging.info("Errors: %s", error_count)

        if results:
            logging.info("Sample classification result:")
            sample = results[0]
            if "llm_classification" in sample and "error" not in sample["llm_classification"]:
                logging.info(json.dumps(sample["llm_classification"], indent=2))

    except Exception as exc:
        logging.error("Fatal error during classification: %s", exc)
        raise

if __name__ == '__main__':
    main()
