"""DBLP source adapter for citation validation."""

from __future__ import annotations

from typing import Optional

from ...parser.dblp_parser import DblpParser
from .base import SourceMatch


class DBLPSource:
    """Adapter that exposes ``DblpParser`` via the shared source interface."""

    source_name = "dblp"

    def __init__(self, parser: DblpParser):
        self.parser = parser

    def search_by_title(self, title: str, threshold: float = 5.0) -> Optional[SourceMatch]:
        record = self.parser.search_by_title(title, threshold=threshold)
        if not record:
            return None

        return SourceMatch(
            source=self.source_name,
            title=record.get("title", ""),
            authors=record.get("authors", []) or [],
            year=str(record.get("year", "") or ""),
            venue=record.get("venue", "") or "",
            metadata={
                "key": record.get("key", ""),
                "type": record.get("type", ""),
                "url": record.get("url", ""),
                "doi": record.get("doi", ""),
            },
        )
