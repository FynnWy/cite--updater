"""Database sources used by citation validation."""

from .base import CitationSource, SourceMatch
from .dblp_source import DBLPSource
from .acl_source import ACLAnthologySource
from .arxiv_source import ArXivSource

AVAILABLE_SOURCES = ("dblp", "acl", "arxiv")


__all__ = [
    "CitationSource",
    "SourceMatch",
    "DBLPSource",
    "ACLAnthologySource",
    "ArXivSource",
    "AVAILABLE_SOURCES",
]

