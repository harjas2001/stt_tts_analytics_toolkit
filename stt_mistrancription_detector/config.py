"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the STT Mistranscription Detector.

Edit column mappings and thresholds below to match your platform's data export.
Domain vocabulary is loaded from vocab/domain_vocab.txt — see
vocab/domain_vocab.example.txt for format. That file is gitignored and
should never be committed.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Column name mappings ──────────────────────────────────────────────────────
COLUMN_CONFIG = {
    "utterance_col":  os.getenv("UTTERANCE_COL", "Utterance"),
    "intent_col":     os.getenv("INTENT_COL", "Intent"),
    "confidence_col": os.getenv("CONFIDENCE_COL", "Intent Confidence"),
    "session_col":    os.getenv("SESSION_COL", "Session ID"),
}

# ── Thresholds ────────────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 0.6))
MIN_WORD_FREQ            = int(os.getenv("MIN_WORD_FREQ", 5))
MAX_EXAMPLE_UTTERANCES   = int(os.getenv("MAX_EXAMPLE_UTTERANCES", 10))

# ── Domain vocabulary ─────────────────────────────────────────────────────────
# Loaded from vocab/domain_vocab.txt — one term per line, # lines ignored.
# This file is private and gitignored. Copy vocab/domain_vocab.example.txt
# to get started.

_VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "domain_vocab.txt"

if not _VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"\n[config] Domain vocabulary file not found: {_VOCAB_PATH}\n"
        f"  Copy vocab/domain_vocab.example.txt → vocab/domain_vocab.txt\n"
        f"  and populate it with your domain terms."
    )

DOMAIN_VOCAB = {
    line.strip().lower()
    for line in _VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
}
