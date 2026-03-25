"""ACL Anthology source adapter for citation validation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

from .base import SourceMatch

logger = logging.getLogger(__name__)


class ACLAnthologySource:
    """Lookup source backed by ``acl-anthology`` local metadata."""

    source_name = "acl"

    def __init__(self, score_cutoff: float = 70.0):
        try:
            from acl_anthology import Anthology
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Missing dependency 'acl-anthology'. Install with: "
                "pip install -r requirements.txt -r requirements-acl.txt"
            ) from exc

        self.score_cutoff = score_cutoff
        self._cache: Dict[str, Optional[SourceMatch]] = {}
        self._papers: List[object] = []
        self._titles: List[str] = []

        logger.info("Initializing ACL Anthology source index...")
        anthology = Anthology.from_repo()
        for paper in anthology.papers():
            title = (getattr(paper, "title", "") or "").strip()
            if not title:
                continue
            self._papers.append(paper)
            self._titles.append(title)

        logger.info("ACL Anthology source index ready with %s records", len(self._titles))

    def search_by_title(self, title: str, threshold: float = 0.0) -> Optional[SourceMatch]:
        cache_key = self._normalize_title(title)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._titles:
            self._cache[cache_key] = None
            return None

        match = process.extractOne(
            title,
            self._titles,
            scorer=fuzz.token_set_ratio,
            processor=str.lower,
            score_cutoff=self.score_cutoff,
        )
        if not match:
            self._cache[cache_key] = None
            return None

        _, _, idx = match
        paper = self._papers[idx]

        source_match = SourceMatch(
            source=self.source_name,
            title=getattr(paper, "title", "") or "",
            authors=self._extract_authors(paper),
            year=self._extract_year(paper),
            venue=self._extract_venue(paper),
            metadata={"acl_id": getattr(paper, "full_id", "") or ""},
        )
        self._cache[cache_key] = source_match
        return source_match

    @staticmethod
    def _extract_authors(paper: object) -> List[str]:
        authors_out: List[str] = []
        for author in getattr(paper, "authors", []) or []:
            if isinstance(author, str):
                name = author.strip()
                if name:
                    authors_out.append(name)
                continue

            name_attr = getattr(author, "name", None)
            if isinstance(name_attr, str):
                name = name_attr.strip()
                if name:
                    authors_out.append(name)
                continue

            if name_attr is not None:
                for attr_name in ("as_first_last", "as_full", "as_last_first"):
                    candidate = getattr(name_attr, attr_name, None)
                    if callable(candidate):
                        candidate = candidate()
                    if isinstance(candidate, str) and candidate.strip():
                        authors_out.append(candidate.strip())
                        break
                else:
                    rendered = str(name_attr).strip()
                    if rendered and not rendered.startswith("<"):
                        authors_out.append(rendered)
                continue

            rendered = str(author).strip()
            if rendered and not rendered.startswith("<"):
                authors_out.append(rendered)

        return authors_out

    @staticmethod
    def _extract_year(paper: object) -> str:
        year = getattr(paper, "year", "")
        if year:
            return str(year)

        full_id = str(getattr(paper, "full_id", "") or "")
        if len(full_id) >= 4 and full_id[:4].isdigit():
            return full_id[:4]
        return ""

    @staticmethod
    def _extract_venue(paper: object) -> str:
        venue = getattr(paper, "venue", None)
        if isinstance(venue, str):
            return venue

        if venue is not None:
            for attr_name in ("acronym", "name", "title"):
                value = getattr(venue, attr_name, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for attr_name in ("booktitle", "event"):
            value = getattr(paper, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join((title or "").lower().split())
