"""
Text preprocessing utilities shared by all pipelines.

- Sentence / paragraph chunking
- Lightweight cleaning (whitespace, control chars)
- Entity extraction (simple regex-based for legal entities)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single text chunk derived from a document."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Preprocessor:
    """
    Converts raw document dicts into lists of Chunk objects.

    Parameters
    ----------
    chunk_size:    Target characters per chunk (soft limit).
    chunk_overlap: Overlap in characters between consecutive chunks.
    min_chunk_len: Discard chunks shorter than this.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_len: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

        # Simple regex to split on sentence boundaries
        self._sentence_re = re.compile(r"(?<=[.!?])\s+")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, doc: dict) -> list[Chunk]:
        """Return overlapping text chunks for a single document dict."""
        text = self._clean(doc["text"])
        raw_chunks = self._sliding_window(text)
        chunks: list[Chunk] = []
        for i, (start, end) in enumerate(raw_chunks):
            chunk_text = text[start:end].strip()
            if len(chunk_text) < self.min_chunk_len:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['id']}_c{i}",
                    doc_id=doc["id"],
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "label": doc.get("label"),
                        "label_str": doc.get("label_str"),
                        "dataset": doc.get("dataset"),
                        **doc.get("metadata", {}),
                    },
                )
            )
        return chunks

    def process_batch(self, docs: list[dict]) -> list[Chunk]:
        """Chunk a list of document dicts."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.process(doc))
        return all_chunks

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Remove control characters and normalize whitespace."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _sliding_window(self, text: str) -> list[tuple[int, int]]:
        """
        Return (start, end) character index pairs using a sliding window.
        Tries to break on whitespace so words aren't split.
        """
        spans: list[tuple[int, int]] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            # Snap to next whitespace so we don't split mid-word
            if end < n:
                snap = text.rfind(" ", start, end + 1)
                if snap > start:
                    end = snap
            spans.append((start, end))
            if end >= n:
                break
            start = max(start + 1, end - self.chunk_overlap)
        return spans

    # ── Entity extraction (rule-based) ────────────────────────────────────────

    _LEGAL_PATTERNS = {
        "case_citation": re.compile(
            r"\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b"          # e.g. 410 U.S. 113
            r"|\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+\b",  # e.g. Roe v. Wade
            re.IGNORECASE,
        ),
        "statute": re.compile(
            r"\b\d+\s+U\.S\.C\.?\s+[§Ss]?\s*\d+\b"     # e.g. 42 U.S.C. § 1983
            r"|\b[A-Z][a-z]+\s+Act\b",                  # e.g. Privacy Act
            re.IGNORECASE,
        ),
        "article": re.compile(
            r"\bArticle\s+\d+\b"                        # e.g. Article 6
            r"|\bRule\s+\d+\b",                         # e.g. Rule 12(b)
            re.IGNORECASE,
        ),
    }

    @classmethod
    def extract_entities(cls, text: str) -> dict[str, list[str]]:
        """Return a dict of entity_type → list of matched strings."""
        entities: dict[str, list[str]] = {}
        for etype, pat in cls._LEGAL_PATTERNS.items():
            matches = list(dict.fromkeys(pat.findall(text)))  # deduplicate, preserve order
            if matches:
                entities[etype] = matches
        return entities
