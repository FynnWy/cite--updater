"""Reference parser dataset/split/evaluation utilities for GROBID fine-tuning.

This module is designed to be additive to the existing pipeline:
- No existing command has to change.
- If no gold BibTeX data is available, the workflow is skipped cleanly.
- If gold data is available, the module can build reference training pairs,
  create reproducible train/val/test splits (80/10/10 by default), and emit
  macro field-level F1 reports for held-out evaluation.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from rapidfuzz import fuzz
from unidecode import unidecode

from .tei_to_json import parse_tei_file

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("data/grobid_finetune")
DEFAULT_GOLD_BIB_SUBDIR = "gold_bibtex"
DEFAULT_MIN_TITLE_SIMILARITY = 70.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_PAPERS = 3000
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


@dataclass
class AlignmentResult:
    """Alignment output between parsed TEI references and gold BibTeX entries."""

    matched_pairs: List[Dict[str, Any]]
    unmatched_predicted: int
    unmatched_gold: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _strip_latex(text: str) -> str:
    """Best-effort LaTeX cleanup for comparisons and reporting."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^}]*)\})?", r"\1", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("~", " ")
    return _normalize_space(cleaned)


def _normalize_title(text: str) -> str:
    cleaned = unidecode(_strip_latex(text).lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return _normalize_space(cleaned)


def _normalize_generic_field(text: str) -> str:
    cleaned = unidecode(_strip_latex(text).lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return _normalize_space(cleaned)


def _normalize_person_name(text: str) -> str:
    cleaned = unidecode(_strip_latex(text).lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = _normalize_space(cleaned)
    return cleaned


def _parse_author_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [item for item in (str(v).strip() for v in value) if item]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if " and " in raw:
            return [item.strip() for item in raw.split(" and ") if item.strip()]
        if ";" in raw:
            return [item.strip() for item in raw.split(";") if item.strip()]
        return [raw]
    return []


def _normalized_author_signature(value: Any) -> str:
    names = [_normalize_person_name(name) for name in _parse_author_list(value)]
    names = sorted(name for name in names if name)
    return "||".join(names)


def _normalized_year(value: Any) -> str:
    if value is None:
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return match.group(0) if match else ""


def _extract_venue(entry: Dict[str, Any]) -> str:
    for key in ("booktitle", "journal", "venue", "publisher", "series"):
        value = entry.get(key, "")
        if value:
            return str(value)
    return ""


def _find_matching_brace(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _parse_bibtex_fields(field_text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    idx = 0
    n = len(field_text)

    while idx < n:
        while idx < n and field_text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break

        name_start = idx
        while idx < n and (field_text[idx].isalnum() or field_text[idx] in "_-"):
            idx += 1
        field_name = field_text[name_start:idx].strip().lower()
        if not field_name:
            idx += 1
            continue

        while idx < n and field_text[idx].isspace():
            idx += 1
        if idx >= n or field_text[idx] != "=":
            while idx < n and field_text[idx] != ",":
                idx += 1
            continue
        idx += 1

        while idx < n and field_text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        if field_text[idx] == "{":
            end = _find_matching_brace(field_text, idx, "{", "}")
            if end == -1:
                break
            raw_value = field_text[idx + 1 : end]
            idx = end + 1
        elif field_text[idx] == '"':
            idx += 1
            start = idx
            escaped = False
            while idx < n:
                ch = field_text[idx]
                if ch == '"' and not escaped:
                    break
                escaped = (ch == "\\" and not escaped)
                if ch != "\\":
                    escaped = False
                idx += 1
            raw_value = field_text[start:idx]
            idx += 1
        else:
            start = idx
            while idx < n and field_text[idx] not in ",\n\r":
                idx += 1
            raw_value = field_text[start:idx]

        fields[field_name] = _normalize_space(raw_value.strip())

    return fields


def parse_bibtex_file(path: Path) -> List[Dict[str, Any]]:
    """Parse a BibTeX file into normalized reference dictionaries."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries: List[Dict[str, Any]] = []
    idx = 0
    n = len(text)

    while idx < n:
        at = text.find("@", idx)
        if at == -1:
            break
        brace_start = text.find("{", at)
        paren_start = text.find("(", at)

        if brace_start == -1 and paren_start == -1:
            break
        if brace_start == -1 or (paren_start != -1 and paren_start < brace_start):
            open_idx = paren_start
            open_char, close_char = "(", ")"
        else:
            open_idx = brace_start
            open_char, close_char = "{", "}"

        entry_type = text[at + 1 : open_idx].strip().lower()
        end_idx = _find_matching_brace(text, open_idx, open_char, close_char)
        if end_idx == -1:
            break

        body = text[open_idx + 1 : end_idx].strip()
        comma_idx = body.find(",")
        if comma_idx == -1:
            idx = end_idx + 1
            continue

        cite_key = body[:comma_idx].strip()
        field_text = body[comma_idx + 1 :]
        raw_fields = _parse_bibtex_fields(field_text)

        title = raw_fields.get("title", "")
        authors = _parse_author_list(raw_fields.get("author", ""))
        venue = _extract_venue(raw_fields)
        year = _normalized_year(raw_fields.get("year", raw_fields.get("date", "")))

        entries.append(
            {
                "entry_type": entry_type,
                "key": cite_key,
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "raw_fields": raw_fields,
            }
        )
        idx = end_idx + 1

    return entries


def _paper_id_from_tei_path(tei_path: Path) -> str:
    name = tei_path.name
    if name.endswith(".grobid.tei.xml"):
        return name[: -len(".grobid.tei.xml")]
    return tei_path.stem


def _build_bib_stem_index(gold_bib_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for bib in sorted(gold_bib_dir.rglob("*.bib")) + sorted(gold_bib_dir.rglob("*.bibtex")):
        index.setdefault(bib.stem, []).append(bib)
    return index


def _find_bib_for_tei(
    tei_file: Path,
    tei_root: Path,
    gold_bib_dir: Path,
    stem_index: Dict[str, List[Path]],
) -> Optional[Path]:
    rel = tei_file.relative_to(tei_root)
    paper_id = _paper_id_from_tei_path(tei_file)

    preferred_same_rel = [
        gold_bib_dir / rel.parent / f"{paper_id}.bib",
        gold_bib_dir / rel.parent / f"{paper_id}.bibtex",
    ]
    for candidate in preferred_same_rel:
        if candidate.exists():
            return candidate

    candidates = stem_index.get(paper_id, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Ambiguous BibTeX matches for %s (%s candidates); skipping",
            tei_file,
            len(candidates),
        )
    return None


def _align_references(
    predicted_refs: List[Dict[str, Any]],
    gold_refs: List[Dict[str, Any]],
    min_title_similarity: float,
) -> AlignmentResult:
    matched_pairs: List[Dict[str, Any]] = []
    used_gold_indices: set[int] = set()
    unmatched_predicted = 0

    for ref in predicted_refs:
        pred_title = str(ref.get("title", "") or "")
        if not pred_title.strip():
            unmatched_predicted += 1
            continue

        best_idx = -1
        best_score = -1.0
        pred_norm = _normalize_title(pred_title)
        for idx, gold in enumerate(gold_refs):
            if idx in used_gold_indices:
                continue
            gold_title = str(gold.get("title", "") or "")
            if not gold_title.strip():
                continue
            score = float(fuzz.ratio(pred_norm, _normalize_title(gold_title)))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= min_title_similarity:
            used_gold_indices.add(best_idx)
            gold = gold_refs[best_idx]
            matched_pairs.append(
                {
                    "predicted": {
                        "title": pred_title,
                        "authors": _parse_author_list(ref.get("authors", [])),
                        "venue": str(ref.get("venue", "") or ""),
                        "year": _normalized_year(ref.get("year", "")),
                    },
                    "gold": {
                        "title": str(gold.get("title", "") or ""),
                        "authors": _parse_author_list(gold.get("authors", [])),
                        "venue": str(gold.get("venue", "") or ""),
                        "year": _normalized_year(gold.get("year", "")),
                    },
                    "title_similarity": best_score,
                }
            )
        else:
            unmatched_predicted += 1

    unmatched_gold = max(0, len(gold_refs) - len(used_gold_indices))
    return AlignmentResult(
        matched_pairs=matched_pairs,
        unmatched_predicted=unmatched_predicted,
        unmatched_gold=unmatched_gold,
    )


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _metric_key(field: str) -> str:
    return f"{field}_metrics"


def _get_field_signature(field: str, value: Any) -> str:
    if field == "authors":
        return _normalized_author_signature(value)
    if field == "title":
        return _normalize_title(str(value or ""))
    if field == "venue":
        return _normalize_generic_field(str(value or ""))
    if field == "year":
        return _normalized_year(value)
    return _normalize_generic_field(str(value or ""))


def compute_field_level_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute field-level precision/recall/F1 and macro-average F1."""
    fields = ["authors", "title", "venue", "year"]
    counters = {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields}

    for row in records:
        pred = row.get("predicted", {})
        gold = row.get("gold", {})

        for field in fields:
            pred_sig = _get_field_signature(field, pred.get(field, ""))
            gold_sig = _get_field_signature(field, gold.get(field, ""))

            if pred_sig and gold_sig:
                if pred_sig == gold_sig:
                    counters[field]["tp"] += 1
                else:
                    counters[field]["fp"] += 1
                    counters[field]["fn"] += 1
            elif pred_sig and not gold_sig:
                counters[field]["fp"] += 1
            elif gold_sig and not pred_sig:
                counters[field]["fn"] += 1

    report: Dict[str, Any] = {}
    f1_values: List[float] = []
    for field in fields:
        tp = counters[field]["tp"]
        fp = counters[field]["fp"]
        fn = counters[field]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        report[_metric_key(field)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    report["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return report


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

    return (train_count, val_count, test_count)


def register_active_finetuned_config(workspace: Path, config_path: Path) -> Path:
    """Register an externally prepared GROBID config as the active fine-tuned profile."""
    if not config_path.exists():
        raise FileNotFoundError(f"Fine-tuned config not found: {config_path}")
    model_dir = workspace / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    active_path = model_dir / "active_grobid_config.json"
    active_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return active_path


def resolve_effective_config_path(requested_config_path: str, workspace: Path) -> str:
    """
    Resolve the config path to use for GROBID runs.

    Priority:
    1) explicit existing requested path
    2) workspace/model/active_grobid_config.json
    3) requested path (even if missing; caller may use default URL fallback)
    """
    requested = Path(requested_config_path)
    if requested.exists():
        return str(requested)

    active = workspace / "model" / "active_grobid_config.json"
    if active.exists():
        logger.info("Using active fine-tuned GROBID config: %s", active)
        return str(active)

    return requested_config_path


def run_reference_parser_pipeline(
    *,
    tei_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    gold_bib_dir: Optional[Path] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_papers: int = DEFAULT_MAX_PAPERS,
    min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    trainer_command: Optional[str] = None,
    trainer_timeout_sec: int = 3600,
    hardware_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build dataset/splits/evaluation reports for reference parser fine-tuning.

    This function is intentionally fault-tolerant: it returns a summary with
    ``status=skipped`` when required inputs are missing.
    """
    workspace = Path(workspace)
    gold_dir = Path(gold_bib_dir) if gold_bib_dir else workspace / DEFAULT_GOLD_BIB_SUBDIR
    dataset_dir = workspace / "dataset"
    splits_dir = dataset_dir / "splits"
    reports_dir = workspace / "reports"
    runs_dir = workspace / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_meta: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "tei_root": str(tei_root),
        "workspace": str(workspace),
        "gold_bib_dir": str(gold_dir),
        "random_seed": random_seed,
        "max_papers": max_papers,
        "min_title_similarity": min_title_similarity,
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hardware_notes": hardware_notes or "",
        },
        "status": "unknown",
    }

    if not tei_root.exists():
        run_meta["status"] = "skipped"
        run_meta["reason"] = f"TEI root not found: {tei_root}"
        _write_pipeline_metadata(run_meta, runs_dir)
        return run_meta

    if not gold_dir.exists():
        run_meta["status"] = "skipped"
        run_meta["reason"] = f"Gold BibTeX directory not found: {gold_dir}"
        _write_pipeline_metadata(run_meta, runs_dir)
        return run_meta

    tei_files = sorted(tei_root.rglob("*.grobid.tei.xml"))
    if not tei_files:
        run_meta["status"] = "skipped"
        run_meta["reason"] = f"No TEI files found under: {tei_root}"
        _write_pipeline_metadata(run_meta, runs_dir)
        return run_meta

    if max_papers > 0 and len(tei_files) > max_papers:
        rng = random.Random(random_seed)
        tei_files = sorted(rng.sample(tei_files, max_papers))

    bib_stem_index = _build_bib_stem_index(gold_dir)
    all_rows: List[Dict[str, Any]] = []
    paper_count = 0
    matched_papers = 0
    missing_bib_count = 0
    unmatched_predicted_total = 0
    unmatched_gold_total = 0

    for tei_file in tei_files:
        paper_count += 1
        bib_path = _find_bib_for_tei(tei_file, tei_root, gold_dir, bib_stem_index)
        if bib_path is None:
            missing_bib_count += 1
            continue

        try:
            parsed_payload = parse_tei_file(tei_file)
            gold_entries = parse_bibtex_file(bib_path)
        except Exception as exc:
            logger.warning("Skipping %s due to parse error: %s", tei_file, exc)
            continue

        alignment = _align_references(
            predicted_refs=parsed_payload.get("references", []),
            gold_refs=gold_entries,
            min_title_similarity=min_title_similarity,
        )
        if alignment.matched_pairs:
            matched_papers += 1
        unmatched_predicted_total += alignment.unmatched_predicted
        unmatched_gold_total += alignment.unmatched_gold

        paper_id = _paper_id_from_tei_path(tei_file)
        for idx, pair in enumerate(alignment.matched_pairs, start=1):
            all_rows.append(
                {
                    "paper_id": paper_id,
                    "pair_id": f"{paper_id}::r{idx}",
                    "tei_path": str(tei_file),
                    "bib_path": str(bib_path),
                    "title_similarity": pair["title_similarity"],
                    "predicted": pair["predicted"],
                    "gold": pair["gold"],
                }
            )

    pairs_path = dataset_dir / "reference_pairs.jsonl"
    total_pairs = _write_jsonl(pairs_path, all_rows)

    run_meta["dataset"] = {
        "tei_files_considered": len(tei_files),
        "papers_processed": paper_count,
        "papers_with_bib_match": matched_papers,
        "missing_bib_count": missing_bib_count,
        "matched_pairs": total_pairs,
        "unmatched_predicted_references": unmatched_predicted_total,
        "unmatched_gold_references": unmatched_gold_total,
        "pairs_file": str(pairs_path),
    }

    if total_pairs == 0:
        run_meta["status"] = "skipped"
        run_meta["reason"] = "No matched reference pairs found; cannot build splits/evaluation."
        _write_pipeline_metadata(run_meta, runs_dir)
        return run_meta

    rng = random.Random(random_seed)
    shuffled_rows = list(all_rows)
    rng.shuffle(shuffled_rows)
    train_count, val_count, test_count = _split_counts(
        total_pairs, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_rows = shuffled_rows[:train_count]
    val_rows = shuffled_rows[train_count : train_count + val_count]
    test_rows = shuffled_rows[train_count + val_count : train_count + val_count + test_count]

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "validation.jsonl"
    test_path = splits_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    evaluation = compute_field_level_metrics(test_rows)
    evaluation.update(
        {
            "evaluated_split": "test",
            "test_size": len(test_rows),
            "generated_at_utc": _utc_now_iso(),
        }
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_path = reports_dir / "reference_parser_evaluation.json"
    eval_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta["splits"] = {
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "test_size": len(test_rows),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "test_file": str(test_path),
    }
    run_meta["evaluation"] = {
        "macro_f1": evaluation.get("macro_f1", 0.0),
        "evaluation_file": str(eval_path),
    }

    trainer_result: Dict[str, Any] = {
        "status": "not_run",
        "reason": "No trainer command provided",
    }
    if trainer_command:
        trainer_result = _run_external_trainer(
            trainer_command=trainer_command,
            timeout_sec=trainer_timeout_sec,
            workspace=workspace,
        )
    run_meta["trainer"] = trainer_result

    run_meta["status"] = "completed"
    _write_pipeline_metadata(run_meta, runs_dir)
    return run_meta


def _run_external_trainer(
    *,
    trainer_command: str,
    timeout_sec: int,
    workspace: Path,
) -> Dict[str, Any]:
    run_log = workspace / "runs" / "trainer_stdout.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    args = shlex.split(trainer_command)
    if not args:
        return {"status": "not_run", "reason": "Empty trainer command"}

    try:
        completed = subprocess.run(
            args,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        run_log.write_text(completed.stdout or "", encoding="utf-8")
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "command": trainer_command,
            "stdout_log": str(run_log),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "reason": f"Trainer command timed out after {timeout_sec}s",
            "command": trainer_command,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "command": trainer_command,
        }


def _write_pipeline_metadata(metadata: Dict[str, Any], runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    latest_path = runs_dir / "latest_reference_pipeline_run.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_path = runs_dir / f"reference_pipeline_run_{stamp}.json"
    payload = json.dumps(metadata, indent=2, ensure_ascii=False)
    latest_path.write_text(payload, encoding="utf-8")
    archived_path.write_text(payload, encoding="utf-8")
