"""Convert GROBID TEI files into JSON records for validation."""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _normalize_scalar(value: str) -> str:
    """Normalize plain text fields to a single-space representation."""
    return " ".join((value or "").split()).strip()


def _parse_person_name(author: ET.Element) -> str:
    pers = author.find("tei:persName", NS)
    if pers is None:
        return ""

    parts = []
    for forename in pers.findall("tei:forename", NS):
        value = _text(forename)
        if value:
            parts.append(value)
    surname = _text(pers.find("tei:surname", NS))
    if surname:
        parts.append(surname)
    return " ".join(parts).strip()


def _extract_reference_venue(monogr: ET.Element | None) -> str:
    """
    Extract a venue string from a TEI ``monogr`` block.

    Priority is journal/proceedings/series title, then event, then publisher.
    """
    if monogr is None:
        return ""

    title_paths = (
        "tei:title[@level='j']",
        "tei:title[@level='m']",
        "tei:title[@level='s']",
    )
    for path in title_paths:
        value = _normalize_scalar(_text(monogr.find(path, NS)))
        if value:
            return value

    for path in ("tei:meeting", "tei:imprint/tei:publisher"):
        value = _normalize_scalar(_text(monogr.find(path, NS)))
        if value:
            return value

    return ""


def _extract_main_biblio(root: ET.Element) -> dict[str, Any]:
    analytic = root.find(".//tei:sourceDesc/tei:biblStruct/tei:analytic", NS)
    title = ""
    authors: list[str] = []

    if analytic is not None:
        title = _text(analytic.find("tei:title[@type='main']", NS))
        if not title:
            title = _text(analytic.find("tei:title[@level='a']", NS))

        for author in analytic.findall("tei:author", NS):
            name = _parse_person_name(author)
            if name:
                authors.append(name)

    if not title:
        title = _text(root.find(".//tei:titleStmt/tei:title", NS))

    date_elem = root.find(
        ".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date", NS
    )
    publication_date = date_elem.get("when", "") if date_elem is not None else ""

    publication_year = None
    if publication_date:
        match = re.match(r"(\d{4})", publication_date)
        if match:
            publication_year = int(match.group(1))

    return {
        "title": title,
        "authors": authors,
        "publication_date": publication_date,
        "publication_year": publication_year,
        "publisher": "",
        "abstract": [],
    }


def _extract_reference(bibl: ET.Element, fallback_id: int) -> dict[str, Any]:
    analytic = bibl.find("tei:analytic", NS)
    monogr = bibl.find("tei:monogr", NS)

    title = ""
    authors: list[str] = []
    if analytic is not None:
        title = _text(analytic.find("tei:title[@type='main']", NS))
        if not title:
            title = _text(analytic.find("tei:title[@level='a']", NS))
        for author in analytic.findall("tei:author", NS):
            name = _parse_person_name(author)
            if name:
                authors.append(name)

    if not title and monogr is not None:
        title = _text(monogr.find("tei:title[@level='m']", NS))
        if not title:
            title = _text(monogr.find("tei:title[@level='j']", NS))

    if not authors and monogr is not None:
        for author in monogr.findall("tei:author", NS):
            name = _parse_person_name(author)
            if name:
                authors.append(name)

    identifiers: dict[str, str] = {}
    for idno in bibl.findall(".//tei:idno", NS):
        key = idno.get("type", "unknown")
        value = _text(idno)
        if value:
            identifiers[key] = value

    ref: dict[str, Any] = {
        "id": bibl.get(XML_ID, f"b{fallback_id}"),
        "title": title,
        "authors": authors,
        "venue": _extract_reference_venue(monogr),
        "year": "",
    }

    date_elem = bibl.find(".//tei:imprint/tei:date", NS)
    date_text = date_elem.get("when", "") if date_elem is not None else _text(date_elem)
    year_match = re.search(r"\b(19|20)\d{2}\b", date_text)
    if year_match:
        ref["year"] = int(year_match.group(0))

    vol = bibl.find(".//tei:biblScope[@unit='volume']", NS)
    volume = _text(vol)
    if volume:
        ref["volume"] = volume

    pages = bibl.find(".//tei:biblScope[@unit='page']", NS)
    if pages is not None:
        from_page = pages.get("from", "").strip()
        to_page = pages.get("to", "").strip()
        if from_page:
            ref["page_start"] = from_page
        if to_page:
            ref["page_end"] = to_page

    if identifiers:
        ref["identifiers"] = identifiers

    return ref


def parse_tei_file(tei_path: Path) -> dict[str, Any]:
    tree = ET.parse(tei_path)
    root = tree.getroot()

    references: list[dict[str, Any]] = []
    for idx, bibl in enumerate(
        root.findall(".//tei:text//tei:listBibl//tei:biblStruct", NS), start=1
    ):
        ref = _extract_reference(bibl, idx)
        if ref.get("title") or ref.get("authors"):
            references.append(ref)

    return {
        "biblio": _extract_main_biblio(root),
        "references": references,
    }


def export_tei_tree_to_json(
    tei_root: Path, json_root: Path, pattern: str = "**/*.grobid.tei.xml"
) -> tuple[int, int]:
    tei_files = sorted(tei_root.glob(pattern))
    if not tei_files:
        logger.warning("No TEI files found for JSON export in %s", tei_root)
        return (0, 0)

    json_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed = 0
    total = len(tei_files)
    logger.info("Exporting %s TEI files to JSON in %s", total, json_root)

    for idx, tei_file in enumerate(tei_files, start=1):
        rel = tei_file.relative_to(tei_root)
        if rel.name.endswith(".grobid.tei.xml"):
            out_name = rel.name[: -len(".grobid.tei.xml")] + ".json"
            json_rel = rel.with_name(out_name)
        else:
            json_rel = rel.with_suffix(".json")
        json_path = json_root / json_rel
        json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = parse_tei_file(tei_file)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            processed += 1
            if idx % 100 == 0 or idx == total:
                logger.info("TEI->JSON progress: %s/%s", idx, total)
        except Exception as exc:
            failed += 1
            logger.error("Failed to convert %s: %s", tei_file, exc)

    logger.info(
        "TEI->JSON export completed (processed=%s, failed=%s, output_dir=%s)",
        processed,
        failed,
        json_root,
    )
    return (processed, failed)
