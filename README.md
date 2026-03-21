# Academic Paper Processing Toolkit

Two-stage pipeline aligned with the paper methodology:
- **Stage 1 - Citation Extraction**: fetch PDFs, run GROBID full-text (PDF -> TEI), convert TEI to TSV/CSV.
- **Stage 2 - Name Matching**: compare extracted citations against authoritative databases (e.g., DBLP) and optionally classify mismatches with an LLM.

All commands are exposed via one CLI: `python -m src.pipeline ...`.

## Step-by-step Setup
Follow this sequence once, then run the stage commands as needed.

### 1) Create your Python environment(s)
Use two environments:
- `.venv` (Python 3.11) for Stage 1 (`download`, `grobid`, `parse`, `to-json`) and optional LLM classification.
- `.venv310` (Python 3.10) for Stage 2 validation with `retriv`.

Windows:
```bash
py -3.11 -m venv .venv
. .venv/Scripts/activate
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Optional extras:
```bash
# ACL Anthology downloader support
pip install -r requirements.txt -r requirements-acl.txt

# LLM classification support
pip install -r requirements.txt -r requirements-llm.txt
```

Backend behavior for classification:
- Linux: uses `vllm` when available.
- macOS/Windows: automatically falls back to `transformers`.

Create `.venv310` for validation:

Windows:
```bash
py -3.10 -m venv .venv310
. .venv310/Scripts/activate
pip install -r requirements.txt
pip install retriv
python -c "from retriv import SparseRetriever; print('retriv ok')"
```

macOS/Linux:
```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install -r requirements.txt
pip install retriv
python -c "from retriv import SparseRetriever; print('retriv ok')"
```

### 2) Initialize DBLP data
Download the DBLP XML dump (required for Stage 2 validation):
```bash
mkdir -p data
curl -L -o data/dblp.xml.gz https://dblp.org/xml/dblp.xml.gz
gunzip -f data/dblp.xml.gz
```

### 3) Optional: scrape DBLP conference pages
Basic run:
```bash
python -m src.citation_extraction.dblp_scraper --output-dir data/dblp_conferences
```

Year-filtered run:
```bash
python -m src.citation_extraction.dblp_scraper \
  --start-year 2015 \
  --end-year 2025 \
  --output-dir data/dblp_conferences \
  --delay 2.0
```

### 4) Stage 1: Citation extraction
#### 4.1 Download PDFs
arXiv only:
```bash
python -m src.pipeline download --source arxiv --output-dir data/arxiv_pdfs
```

ACL only (titles already on arXiv are skipped by default):
```bash
python -m src.pipeline download --source acl --acl-output-dir data/acl_pdfs
```

Both sources in one command:
```bash
python -m src.pipeline download --source both --output-dir data/arxiv_pdfs --acl-output-dir data/acl_pdfs
```

#### 4.2 Run GROBID (only needed for this step)
Install Docker Desktop if needed (example for macOS/Homebrew):
```bash
brew install --cask docker
```
Then open Docker Desktop once so the daemon is running.

Terminal A (start GROBID server):
```bash
source .venv/bin/activate
docker run --rm --init -p 8070:8070 grobid/grobid:0.8.0
```

Terminal B (run pipeline against local GROBID):
```bash
source .venv/bin/activate
unset GROBID_URL
export GROBID_URL=http://localhost:8070/api
curl -sS http://localhost:8070/api/isalive
python -m src.pipeline grobid --input-dir data/arxiv_pdfs --output-dir data/outputs/arxiv_pdfs
```

#### 4.3 Parse TEI XML to tabular metadata
```bash
python -m src.pipeline parse \
  --input-dir data/outputs/arxiv_pdfs \
  --pattern "**/*.grobid.tei.xml" \
  --output-csv data/arxiv_metadata.csv
```

#### 4.4 Optional: export TEI XML to JSON manually
`grobid` already exports JSON by default. Run this only if you skipped JSON export or want to regenerate it:
```bash
python -m src.pipeline to-json --input-dir data/outputs/arxiv_pdfs --output-dir data/parsed_jsons
```

### 5) Stage 2: Validate citations against DBLP
Activate `.venv310` (created in Step 1):
```bash
source .venv310/bin/activate
```

Run validation:
```bash
python -m src.pipeline validate --input-dir data/parsed_jsons --dblp-xml data/dblp.xml --output-dir validation_results
```

### 6) Optional: LLM mismatch classification
Use `.venv` (or another environment with `requirements-llm.txt` installed):
```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt -r requirements-llm.txt
python -m src.pipeline classify \
  --input_file validation_results/validation_results.json \
  --output_file validation_results/classified_results.json
```

Force the transformers backend explicitly:
```bash
python -m src.pipeline classify --backend transformers --transformers_device auto
```

## Pipeline overview
Stage 1 - Citation extraction (requires running GROBID for `grobid`)
- `download` - unified command: `src.models.arxiv_fetcher` + `src.models.acl_fetcher` (`--source arxiv|acl|both`)
- `grobid` - `src.citation_extraction.grobid_runner`
- `parse` - `src.citation_extraction.grobid_parser`

Stage 2 - Name matching (no GROBID needed)
- `validate` - `src.name_matching.validate_citations`
- `classify` (optional) - `src.models.vllm_classifier`

![Reference checking pipeline](docs/refcheck_updated.png)

## Repository map (roles)
```
src/pipeline.py                     # CLI entrypoint
src/citation_extraction/            # Stage 1: PDF -> TEI/TSV
  grobid_runner.py
  grobid_parser.py
  dblp_scraper.py
src/name_matching/                  # Stage 2: DB lookups & validation
  api_caller.py
  analyze_matches.py
  validate_citations.py
  semantic_scholar.py
src/models/                         # Optional LLM classification
  acl_fetcher.py
  arxiv_fetcher.py
  prompt.py
  vllm_classifier.py
src/parser/dblp_parser.py           # Shared DBLP utilities
examples/                           # Demos & sample outputs
docs/refcheck_updated.png           # Pipeline graphic (inline preview)
docs/refcheck_updated.pdf           # High-res PDF version of the pipeline graphic
validation_results/                 # Outputs written by validate/classify
requirements.txt
requirements-acl.txt               # Optional deps for ACL downloader
requirements-llm.txt               # Optional deps for classify (vLLM/transformers)
```

## Core commands & scripts
- `python -m src.pipeline download --source arxiv` - download PDFs from arXiv using DBLP metadata (fuzzy matching, resumable).
- `python -m src.pipeline download --source acl` - download ACL Anthology PDFs (default: skip titles already available on arXiv; needs `requirements-acl.txt`).
- `python -m src.pipeline download --source both` - run both download sources in one command.
- `python -m src.pipeline grobid` - run GROBID full-text over PDFs (needs running GROBID service).
- `python -m src.pipeline parse` - convert TEI XML to TSV/CSV.
- `python -m src.pipeline validate` - match citations against DBLP; produces `validation_results.json`.
- `python -m src.pipeline classify` - optional LLM-based classification of mismatches (auto backend selection: vLLM on Linux, transformers fallback otherwise).

## Examples
- `examples/api_caller_demo.py` - multi-source search (DBLP/arXiv/Semantic Scholar); writes `api_caller_sample*.json`.
- `examples/dblp_parser_demo.py` - small DBLP lookups against a local XML; writes `dblp_parser_sample.json`.

## Configuration
- **GROBID**: pass server URL/port via environment or grobid_client defaults; if you keep a config file, point `--config-path` to it (default `config/config.json` if you create one). Start GROBID, e.g.  
  `docker run --rm --init -p 8070:8070 grobid/grobid:0.8.0`
- **Semantic Scholar API key**: set `SEMANTIC_SCHOLAR_API_KEY` in `.env` (copy `.env.example`).

## Dependencies
- Core: arxiv, requests, fuzzywuzzy (+python-Levenshtein), nameparser, python-dotenv, tqdm, rapidfuzz, unidecode, retriv, grobid-client, pandas, numpy, matplotlib.
- Optional (ACL download, install via `requirements-acl.txt`): acl-anthology.
- Optional (LLM, install via `requirements-llm.txt`): torch, huggingface_hub, transformers (<5), vllm (Linux-only).

## Notes & limitations
- GROBID is only needed for the `grobid` step; later steps work on existing TEI/TSV/JSON data.
- On Python 3.11, `acl-anthology` may conflict with `grobid-client` (upstream dependency constraints); if needed, run ACL download in a separate environment.
- On macOS, `requirements-llm.txt` pins `numpy<2` for compatibility with torch 2.2.x wheels.
- API rate limits (arXiv, DBLP, Semantic Scholar) apply; built-in throttling is basic.
- Large runs (PDF download + GROBID) can be slow—run per conference/year if needed.
