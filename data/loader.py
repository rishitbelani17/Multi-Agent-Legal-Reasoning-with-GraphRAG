"""
Dataset loaders for LEDGAR, CaseHOLDER, and ECtHR.

Each loader returns a list of dicts with a common schema:
  {
    "id":        str,           # unique document/example identifier
    "text":      str,           # full document / premise text
    "label":     str | int,     # ground-truth label
    "label_str": str,           # human-readable label string
    "metadata":  dict,          # any additional fields
    "dataset":   str,           # "ledgar" | "caseholder" | "ecthr"
  }
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import config

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _seed_sample(items: list, k: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(items, min(k, len(items)))


# ── LEDGAR ────────────────────────────────────────────────────────────────────

def load_ledgar(
    subset_size: int = config.LEDGAR_SUBSET_SIZE,
    seed: int = config.RANDOM_SEED,
    cache_dir: str = config.DATA_CACHE_DIR,
) -> list[dict[str, Any]]:
    """
    Load LEDGAR contract clause classification dataset.
    HuggingFace: lex_glue / ledgar  (binary verdict → multi-class clause type)

    Labels are clause category strings (e.g. "Representations", "Indemnification").
    We treat this as a multi-class classification task.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Loading LEDGAR dataset …")
    ds = load_dataset("lex_glue", "ledgar", cache_dir=cache_dir, trust_remote_code=True)

    # Merge train + validation splits so we have enough examples
    split = ds["test"] if "test" in ds else ds["train"]
    label_names: list[str] = split.features["label"].names

    records: list[dict] = []
    for row in split:
        records.append(
            {
                "id": f"ledgar_{len(records)}",
                "text": row["text"],
                "label": row["label"],
                "label_str": label_names[row["label"]],
                "metadata": {},
                "dataset": "ledgar",
            }
        )

    sampled = _seed_sample(records, subset_size, seed)
    logger.info("LEDGAR: %d examples loaded (subset of %d)", len(sampled), len(records))
    return sampled


# ── CaseHOLDER ────────────────────────────────────────────────────────────────

def load_caseholder(
    subset_size: int = config.CASEHOLDER_SUBSET_SIZE,
    seed: int = config.RANDOM_SEED,
    cache_dir: str = config.DATA_CACHE_DIR,
) -> list[dict[str, Any]]:
    """
    Load CaseHOLD dataset (case-holding classification).
    HuggingFace: casehold/casehold

    Format: multiple-choice – given a case text, pick the correct holding
    from 5 candidates.  Label is 0-4 (index of correct answer).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Loading CaseHOLDER (CaseHOLD) dataset …")
    ds = load_dataset("casehold/casehold", "all", cache_dir=cache_dir, trust_remote_code=True)

    split = ds["test"] if "test" in ds else ds["train"]

    records: list[dict] = []
    for row in split:
        # Build a readable representation of the MC question
        choices = [
            row.get(f"holding_{i}", "") for i in range(5)
        ]
        text = (
            row.get("citing_prompt", "")
            + "\n\nChoices:\n"
            + "\n".join(f"({i}) {c}" for i, c in enumerate(choices))
        )
        label = int(row.get("label", 0))
        records.append(
            {
                "id": f"caseholder_{len(records)}",
                "text": text,
                "label": label,
                "label_str": f"holding_{label}",
                "metadata": {
                    "choices": choices,
                    "citing_prompt": row.get("citing_prompt", ""),
                },
                "dataset": "caseholder",
            }
        )

    sampled = _seed_sample(records, subset_size, seed)
    logger.info(
        "CaseHOLDER: %d examples loaded (subset of %d)", len(sampled), len(records)
    )
    return sampled


# ── ECtHR ─────────────────────────────────────────────────────────────────────

def load_ecthr(
    subset_size: int = config.ECTHR_SUBSET_SIZE,
    seed: int = config.RANDOM_SEED,
    cache_dir: str = config.DATA_CACHE_DIR,
) -> list[dict[str, Any]]:
    """
    Load ECtHR (European Court of Human Rights) dataset.
    HuggingFace: lex_glue / ecthr_a  (binary violation prediction per article)

    Label: list of violated articles (multi-label); we binarize to
    violated (1) / not violated (0) based on whether any article is flagged.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Loading ECtHR dataset …")
    ds = load_dataset("lex_glue", "ecthr_a", cache_dir=cache_dir, trust_remote_code=True)
    split = ds["test"] if "test" in ds else ds["train"]

    records: list[dict] = []
    for row in split:
        # Flatten list of facts into a single text block
        facts = row.get("text", [])
        if isinstance(facts, list):
            text = " ".join(facts)
        else:
            text = str(facts)

        labels = row.get("labels", [])
        binary_label = int(len(labels) > 0)
        records.append(
            {
                "id": f"ecthr_{len(records)}",
                "text": text,
                "label": binary_label,
                "label_str": "violated" if binary_label else "not_violated",
                "metadata": {"articles": labels},
                "dataset": "ecthr",
            }
        )

    sampled = _seed_sample(records, subset_size, seed)
    logger.info("ECtHR: %d examples loaded (subset of %d)", len(sampled), len(records))
    return sampled


# ── Unified loader ────────────────────────────────────────────────────────────

class DataLoader:
    """Convenience wrapper that loads one or more datasets."""

    SUPPORTED = {"ledgar", "caseholder", "ecthr"}

    def __init__(
        self,
        datasets: list[str] | None = None,
        cache_dir: str = config.DATA_CACHE_DIR,
        seed: int = config.RANDOM_SEED,
    ):
        self.datasets = datasets or ["ledgar", "caseholder"]
        self.cache_dir = cache_dir
        self.seed = seed
        os.makedirs(cache_dir, exist_ok=True)

    def load(self, name: str) -> list[dict]:
        name = name.lower()
        if name not in self.SUPPORTED:
            raise ValueError(f"Unknown dataset '{name}'. Choose from {self.SUPPORTED}")
        if name == "ledgar":
            return load_ledgar(cache_dir=self.cache_dir, seed=self.seed)
        if name == "caseholder":
            return load_caseholder(cache_dir=self.cache_dir, seed=self.seed)
        if name == "ecthr":
            return load_ecthr(cache_dir=self.cache_dir, seed=self.seed)

    def load_all(self) -> dict[str, list[dict]]:
        return {name: self.load(name) for name in self.datasets}
