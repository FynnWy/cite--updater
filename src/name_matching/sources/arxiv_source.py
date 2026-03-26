"""arXiv source adapter for citation validation."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import arxiv
from rapidfuzz import fuzz

from .base import SourceMatch

logger = logging.getLogger(__name__)


class ArXivSource:
    """Lookup source backed by the official ``arxiv`` client."""

    source_name = "arxiv"

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.client = arxiv.Client(page_size=max_results, delay_seconds=0, num_retries=3)
        self._cache: Dict[str, Optional[SourceMatch]] = {}

    def search_by_title(self, title: str, threshold: float = 0.0) -> Optional[SourceMatch]:
        cache_key = self._normalize_title(title)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            search = arxiv.Search(
                query=f'ti:"{title}"',
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            best_match: Optional[SourceMatch] = None
            best_similarity = -1.0

            for result in self.client.results(search):
                similarity = fuzz.ratio(self._normalize_title(title), self._normalize_title(result.title))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = SourceMatch(
                        source=self.source_name,
                        title=result.title or "",
                        authors=[author.name for author in (result.authors or []) if getattr(author, "name", "")],
                        year=str(getattr(getattr(result, "published", None), "year", "") or ""),
                        venue="arXiv",
                        metadata={
                            "arxiv_id": result.get_short_id(),
                            "entry_id": getattr(result, "entry_id", ""),
                        },
                    )

            self._cache[cache_key] = best_match
            return best_match
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("arXiv lookup failed for title '%s': %s", title[:120], exc)
            self._cache[cache_key] = None
            return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join((title or "").lower().split())
