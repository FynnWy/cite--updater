"""Common interfaces for citation lookup sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class SourceMatch:
    """Normalized citation match returned by a lookup source."""

    source: str
    title: str
    authors: List[str]
    year: str = ""
    venue: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CitationSource(Protocol):
    """Protocol implemented by all citation lookup sources."""

    source_name: str

    def search_by_title(self, title: str, threshold: float = 0.0) -> Optional[SourceMatch]:
        """Return the best match for a title or ``None`` if no match exists."""
        ...
