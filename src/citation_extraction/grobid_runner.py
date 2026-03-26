"""Thin wrapper around grobid_client to process PDF batches."""
import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
import httpx

try:
    # Legacy grobid-client API (<0.1)
    from grobid_client.grobid_client import GrobidClient as LegacyGrobidClient
except ModuleNotFoundError:  # pragma: no cover - depends on installed grobid-client
    LegacyGrobidClient = None

logger = logging.getLogger(__name__)


def _resolve_base_url(config_path: str) -> str:
    """Resolve GROBID base URL from env or optional JSON config."""
    if os.getenv("GROBID_URL"):
        url = os.environ["GROBID_URL"]
        parsed = urlparse(url)
        if parsed.path in ("", "/"):
            return f"{url.rstrip('/')}/api"
        return url

    cfg = Path(config_path)
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            server = data.get("grobid_server")
            port = data.get("grobid_port")
            if server and port:
                base = f"http://{server}:{port}"
                parsed = urlparse(base)
                if parsed.path in ("", "/"):
                    return f"{base}/api"
                return base
            if isinstance(server, str) and server.startswith("http"):
                parsed = urlparse(server)
                if parsed.path in ("", "/"):
                    return f"{server.rstrip('/')}/api"
                return server
        except Exception:
            logger.warning("Could not parse config file at %s", cfg)

    return "http://localhost:8070/api"


def _run_grobid_legacy(
    *,
    in_path: Path,
    out_path: Path,
    config_path: str,
    verbose: bool,
    force: bool,
) -> None:
    if LegacyGrobidClient is None:
        raise RuntimeError("Legacy grobid-client API is not available")

    client = LegacyGrobidClient(config_path=config_path)
    client.process(
        "processFulltextDocument",
        str(in_path),
        output=str(out_path),
        verbose=verbose,
        json_output=True,
        force=force,
    )


def _run_grobid_openapi(
    *,
    in_path: Path,
    out_path: Path,
    config_path: str,
    force: bool,
) -> None:
    from grobid_client.api.pdf import process_fulltext_document
    from grobid_client.client import Client
    from grobid_client.models.process_form import ProcessForm
    from grobid_client.types import File

    base_url = _resolve_base_url(config_path)
    health_url = f"{base_url.rstrip('/')}/isalive"
    try:
        health = httpx.get(health_url, timeout=5.0)
        if health.status_code != 200 or health.text.strip().lower() != "true":
            raise RuntimeError(
                f"GROBID health check failed at {health_url} "
                f"(status={health.status_code}, body={health.text[:120]!r})"
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach GROBID at {health_url}. "
            "Start/restart the server and verify GROBID_URL."
        ) from exc

    client = Client(base_url=base_url, timeout=120.0)
    pdf_files = sorted(in_path.rglob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", in_path)
        return

    logger.info("Found %s PDF files to process (base_url=%s)", len(pdf_files), base_url)

    processed = 0
    skipped = 0
    failed = 0

    total = len(pdf_files)
    for idx, pdf in enumerate(pdf_files, start=1):
        rel = pdf.relative_to(in_path)
        out_file = out_path / rel.with_suffix(".grobid.tei.xml")
        if out_file.exists() and not force:
            skipped += 1
            logger.info("[%s/%s] skipped existing: %s", idx, total, rel)
            continue

        logger.info("[%s/%s] processing: %s", idx, total, rel)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with pdf.open("rb") as fh:
            form = ProcessForm(
                input_=File(
                    payload=fh,
                    file_name=pdf.name,
                    mime_type="application/pdf",
                )
            )
            try:
                response = process_fulltext_document.sync_detailed(
                    client=client,
                    multipart_data=form,
                )
            except httpx.HTTPError as exc:
                failed += 1
                logger.error("GROBID connection failed for %s: %s", pdf, exc)
                continue

        if response.status_code >= 400:
            failed += 1
            logger.error(
                "GROBID failed for %s (status=%s): %s",
                pdf,
                response.status_code,
                response.content[:500].decode("utf-8", errors="ignore"),
            )
            continue

        out_file.write_bytes(response.content)
        processed += 1

    logger.info(
        "GROBID completed (processed=%s, skipped=%s, failed=%s, base_url=%s)",
        processed,
        skipped,
        failed,
        base_url,
    )


def run_grobid(
    input_dir: str = "data/arxiv_pdfs",
    output_dir: str = "data/outputs/arxiv_pdfs",
    config_path: str = "./config/config.json",
    parsed_json_dir: str = "data/parsed_jsons",
    export_json: bool = True,
    verbose: bool = True,
    force: bool = True,
    finetune_workspace: str = "data/grobid_finetune",
    gold_bib_dir: Optional[str] = None,
    run_reference_training_pipeline: bool = True,
    max_finetune_papers: int = 3000,
    random_seed: int = 42,
    finetune_title_match_threshold: float = 70.0,
    trainer_command: Optional[str] = None,
    trainer_timeout_sec: int = 3600,
    hardware_notes: Optional[str] = None,
    finetuned_config_path: Optional[str] = None,
) -> None:
    """Process PDFs through GROBID fulltext service."""
    from src.citation_extraction.tei_to_json import export_tei_tree_to_json
    from src.citation_extraction.reference_parser_training import (
        register_active_finetuned_config,
        resolve_effective_config_path,
        run_reference_parser_pipeline,
    )

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    workspace_path = Path(finetune_workspace)

    if finetuned_config_path:
        try:
            active_config = register_active_finetuned_config(
                workspace=workspace_path,
                config_path=Path(finetuned_config_path),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to register fine-tuned config: {exc}") from exc
        logger.info("Registered fine-tuned GROBID config at %s", active_config)

    effective_config_path = resolve_effective_config_path(config_path, workspace_path)
    if effective_config_path != config_path:
        logger.info(
            "Using resolved GROBID config path: %s (requested: %s)",
            effective_config_path,
            config_path,
        )

    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Starting GROBID: %s -> %s", in_path, out_path)
    if LegacyGrobidClient is not None:
        _run_grobid_legacy(
            in_path=in_path,
            out_path=out_path,
            config_path=effective_config_path,
            verbose=verbose,
            force=force,
        )
    else:
        _run_grobid_openapi(
            in_path=in_path,
            out_path=out_path,
            config_path=effective_config_path,
            force=force,
        )

    if export_json:
        export_tei_tree_to_json(out_path, Path(parsed_json_dir))

    if run_reference_training_pipeline:
        summary = run_reference_parser_pipeline(
            tei_root=out_path,
            workspace=workspace_path,
            gold_bib_dir=Path(gold_bib_dir) if gold_bib_dir else None,
            random_seed=random_seed,
            max_papers=max_finetune_papers,
            min_title_similarity=finetune_title_match_threshold,
            trainer_command=trainer_command,
            trainer_timeout_sec=trainer_timeout_sec,
            hardware_notes=hardware_notes,
        )
        logger.info(
            "Reference parser pipeline status=%s (%s)",
            summary.get("status"),
            summary.get("reason", "ok"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GROBID fulltext on a PDF tree")
    parser.add_argument("--input-dir", default="data/arxiv_pdfs")
    parser.add_argument("--output-dir", default="data/outputs/arxiv_pdfs")
    parser.add_argument("--config-path", default="./config/config.json")
    parser.add_argument("--parsed-json-dir", default="data/parsed_jsons")
    parser.add_argument("--no-export-json", action="store_true")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    parser.add_argument("--finetune-workspace", default="data/grobid_finetune")
    parser.add_argument("--gold-bib-dir", default=None)
    parser.add_argument("--no-reference-training-pipeline", action="store_true")
    parser.add_argument("--max-finetune-papers", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--finetune-title-match-threshold", type=float, default=70.0)
    parser.add_argument("--trainer-command", type=str, default=None)
    parser.add_argument("--trainer-timeout-sec", type=int, default=3600)
    parser.add_argument("--hardware-notes", type=str, default=None)
    parser.add_argument("--finetuned-config-path", type=str, default=None)
    args = parser.parse_args()

    run_grobid(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        parsed_json_dir=args.parsed_json_dir,
        export_json=not args.no_export_json,
        verbose=not args.no_verbose,
        force=not args.no_force,
        finetune_workspace=args.finetune_workspace,
        gold_bib_dir=args.gold_bib_dir,
        run_reference_training_pipeline=not args.no_reference_training_pipeline,
        max_finetune_papers=args.max_finetune_papers,
        random_seed=args.random_seed,
        finetune_title_match_threshold=args.finetune_title_match_threshold,
        trainer_command=args.trainer_command,
        trainer_timeout_sec=args.trainer_timeout_sec,
        hardware_notes=args.hardware_notes,
        finetuned_config_path=args.finetuned_config_path,
    )


if __name__ == "__main__":
    main()
