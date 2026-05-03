"""
STT Mistranscription Detector v2.1
=====================================
Identifies likely mistranscribed words in Genesys Cloud voicebot utterance data
and produces a confusion map: for each domain word, shows WHAT it is being
mistranscribed as, with real example utterances containing each variant.

Usage:
    python stt_mistranscription_detector_v2.1.py --input your_data.csv --output results/

Requirements:
    pip install pandas openpyxl nltk jellyfish tqdm colorama

Changelog v2.1:
    - FIX: Example utterances now show sentences containing the VARIANT word,
           not the target — so "sim" examples show "my slim card won't work" etc.
    - FIX: Examples labelled by which variant triggered them
    - FIX: Style import added (was crashing on console print)
    - FIX: Stopwords filtered in ALL signals, not just Signal 2
    - FIX: Levenshtein threshold now ratio-based to prevent long-word false positives
    - FIX: Confusion map sorted by total variant frequency, not variant count
    - FIX: Domain vocab seeds filtered to only phonetically-risky words (short /
           ambiguous) to reduce false positives from common English words
    - REMOVED: Ground truth signal, channel column
"""

import re
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import jellyfish
import nltk
from nltk.corpus import words as nltk_words
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION — loaded from config.py
# ─────────────────────────────────────────────
from config import (
    COLUMN_CONFIG,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_WORD_FREQ,
    DOMAIN_VOCAB,
    CONFUSION_MAP_SEEDS,
)

MAX_EXAMPLES_PER_VARIANT = 10

# ─────────────────────────────────────────────
#  NLTK SETUP
# ─────────────────────────────────────────────
def setup_nltk():
    for resource, path in [
        ("words",     "corpora/words"),
        ("punkt",     "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK: {resource}")
            nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────
#  DATA LOADING + COLUMN MAPPING
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    print(f"\n{Fore.CYAN}Loading: {p.name}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, engine="openpyxl")
    else:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(p, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
    print(f"  {len(df):,} rows x {len(df.columns)} columns")
    print(f"  Columns detected: {list(df.columns)}")
    return df


def map_columns(df: pd.DataFrame, config: dict) -> dict:
    col_map = {}
    lower = {c.lower(): c for c in df.columns}
    for key, expected in config.items():
        if expected in df.columns:
            col_map[key] = expected
        elif expected.lower() in lower:
            col_map[key] = lower[expected.lower()]
        else:
            hits = [c for c in df.columns
                    if expected.lower().replace("_", "") in c.lower().replace("_", "")]
            if hits:
                col_map[key] = hits[0]
                print(f"  {Fore.YELLOW}'{expected}' not found — using '{hits[0]}'")
            else:
                print(f"  {Fore.YELLOW}WARNING: '{expected}' not found. Some signals will be skipped.")
                col_map[key] = None
    return col_map

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
_STOPWORDS = None

def get_stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(nltk.corpus.stopwords.words("english"))
    return _STOPWORDS


def tokenize(text: str, remove_stopwords: bool = False) -> list:
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^\w\s']", " ", text.lower().strip())
    tokens = [w for w in text.split() if len(w) > 1]
    if remove_stopwords:
        sw = get_stopwords()
        tokens = [w for w in tokens if w not in sw]
    return tokens


def norm_conf(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.5)
    return (s / 100.0 if s.max() > 1.5 else s).clip(0, 1)


def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
        prev = curr
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    """Edit distance as a fraction of the longer word length (0 = identical, 1 = completely different)."""
    dist = levenshtein(a, b)
    return dist / max(len(a), len(b), 1)


# ─────────────────────────────────────────────
#  SIGNAL 1: Low-confidence word frequency
#  Words disproportionately common in low-conf utterances
# ─────────────────────────────────────────────
def signal_low_confidence(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    print(f"\n{Fore.GREEN}[1/4] Low-confidence word frequency...")
    utt  = col_map.get("utterance_col")
    conf = col_map.get("confidence_col")
    if not utt or not conf:
        print("  Skipped — missing utterance or confidence column.")
        return pd.DataFrame()

    df = df.copy()
    df["_c"] = norm_conf(df[conf])
    low  = df[df["_c"] < LOW_CONFIDENCE_THRESHOLD]
    high = df[df["_c"] >= LOW_CONFIDENCE_THRESHOLD]
    print(f"  Low-conf: {len(low):,}  |  High-conf: {len(high):,}")

    # FIX: remove stopwords here too, not just in Signal 2
    lw = Counter(w for t in low[utt]  for w in tokenize(t, remove_stopwords=True))
    hw = Counter(w for t in high[utt] for w in tokenize(t, remove_stopwords=True))
    tl = max(sum(lw.values()), 1)
    th = max(sum(hw.values()), 1)

    rows = []
    for word in set(lw) | set(hw):
        lc, hc = lw.get(word, 0), hw.get(word, 0)
        if lc < MIN_WORD_FREQ:
            continue
        ratio = (lc / tl) / max(hc / th, 1e-9)
        rows.append({
            "word": word,
            "low_conf_count":  lc,
            "high_conf_count": hc,
            "signal_score":    round(ratio, 3),
        })

    out = pd.DataFrame(rows).sort_values("signal_score", ascending=False)
    print(f"  Candidates: {len(out):,}")
    return out


# ─────────────────────────────────────────────
#  SIGNAL 2: Out-of-vocabulary detection
#  Words not in English dictionary or domain vocab
# ─────────────────────────────────────────────
def signal_oov(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    print(f"\n{Fore.GREEN}[2/4] Out-of-vocabulary detection...")
    utt = col_map.get("utterance_col")
    if not utt:
        print("  Skipped — missing utterance column.")
        return pd.DataFrame()

    eng   = set(w.lower() for w in nltk_words.words())
    valid = eng | DOMAIN_VOCAB

    counts = Counter(w for t in df[utt]
                     for w in tokenize(t, remove_stopwords=True))

    rows = []
    for word, cnt in counts.items():
        if cnt < MIN_WORD_FREQ or word in valid or word.isdigit():
            continue
        sdx = jellyfish.soundex(word)
        mph = jellyfish.metaphone(word)
        sim = [v for v in DOMAIN_VOCAB
               if jellyfish.soundex(v) == sdx or jellyfish.metaphone(v) == mph]
        rows.append({
            "word":                   word,
            "frequency":              cnt,
            "soundex":                sdx,
            "metaphone":              mph,
            "phonetically_similar_to": ", ".join(sim[:5]) if sim else "",
        })

    out = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    print(f"  OOV words: {len(out):,}")
    return out


# ─────────────────────────────────────────────
#  SIGNAL 3: Phonetic collision with domain vocab
#  Corpus words that sound like domain terms
# ─────────────────────────────────────────────
def signal_phonetic(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    print(f"\n{Fore.GREEN}[3/4] Phonetic collision with domain vocab...")
    utt = col_map.get("utterance_col")
    if not utt:
        print("  Skipped — missing utterance column.")
        return pd.DataFrame()

    counts = Counter(w for t in df[utt]
                     for w in tokenize(t, remove_stopwords=True))
    dlist  = list(DOMAIN_VOCAB)

    rows = []
    for word, cnt in counts.items():
        if cnt < MIN_WORD_FREQ or word in DOMAIN_VOCAB:
            continue
        hits = []
        for dw in dlist:
            jw       = jellyfish.jaro_winkler_similarity(word, dw)
            phonetic = (jellyfish.soundex(word) == jellyfish.soundex(dw) or
                        jellyfish.metaphone(word) == jellyfish.metaphone(dw))
            if phonetic and jw > 0.75:
                hits.append((dw, round(jw, 3)))
        if hits:
            hits.sort(key=lambda x: -x[1])
            rows.append({
                "word":                  word,
                "frequency":             cnt,
                "possible_intended_word": hits[0][0],
                "similarity_score":       hits[0][1],
                "all_candidates":         ", ".join(f"{w}({s})" for w, s in hits[:5]),
            })

    out = pd.DataFrame(rows).sort_values("similarity_score", ascending=False)
    print(f"  Phonetic candidates: {len(out):,}")
    return out


# ─────────────────────────────────────────────
#  SIGNAL 4: Intent–word anomaly
#  Words correlated with low confidence + intent noise
# ─────────────────────────────────────────────
def signal_intent_anomaly(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    print(f"\n{Fore.GREEN}[4/4] Intent-word anomaly scoring...")
    utt    = col_map.get("utterance_col")
    intent = col_map.get("intent_col")
    conf   = col_map.get("confidence_col")
    if not utt or not intent:
        print("  Skipped — missing utterance or intent column.")
        return pd.DataFrame()

    df = df.copy()
    df["_c"] = norm_conf(df[conf]) if conf else 0.8

    wi = defaultdict(lambda: defaultdict(int))
    wc = defaultdict(list)
    for _, row in df.iterrows():
        words = tokenize(str(row.get(utt, "")), remove_stopwords=True)
        ilab  = str(row.get(intent, "unknown"))
        c     = row.get("_c", 0.8)
        for w in words:
            wi[w][ilab] += 1
            wc[w].append(c)

    rows = []
    for word, idist in wi.items():
        total = sum(idist.values())
        if total < MIN_WORD_FREQ:
            continue
        avg_conf  = np.mean(wc[word])
        n_intents = len(idist)
        top_i     = max(idist, key=idist.get)
        top_pct   = idist[top_i] / total
        score     = (1 - avg_conf) * (1 + n_intents / 10)
        rows.append({
            "word":             word,
            "total_occurrences": total,
            "avg_confidence":   round(avg_conf, 3),
            "num_intents":      n_intents,
            "top_intent":       top_i,
            "top_intent_pct":   round(top_pct, 3),
            "anomaly_score":    round(score, 4),
        })

    out = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False)
    print(f"  Intent anomaly candidates: {len(out):,}")
    return out


# ─────────────────────────────────────────────
#  CONFUSION MAP
#  For each seed word (domain term), find corpus words that are phonetically /
#  orthographically similar — these are the likely mistranscriptions.
#  Examples are pulled from utterances containing the VARIANT, not the target.
# ─────────────────────────────────────────────
def build_confusion_map(df: pd.DataFrame, col_map: dict, seeds: list) -> dict:
    """
    Returns:
        {
          target_word: {
            "variants": { variant_word: count, ... },
            "variant_examples": { variant_word: [utterance, ...], ... }
          }
        }
    """
    print(f"\n{Fore.CYAN}Building confusion map for {len(seeds)} seed words...")
    utt  = col_map.get("utterance_col")
    conf = col_map.get("confidence_col")
    if not utt:
        return {}

    df = df.copy()
    df["_c"] = norm_conf(df[conf]) if conf else 0.8

    # Build per-word corpus stats
    corpus_counts   = Counter()
    # FIX: key examples by the word that APPEARS in the utterance (the variant),
    #      not by what we think the intended word is (the target)
    corpus_examples = defaultdict(list)   # variant_word -> [utterance_text, ...]

    for _, row in df.iterrows():
        utt_text = str(row.get(utt, ""))
        c        = row.get("_c", 0.8)
        words    = tokenize(utt_text, remove_stopwords=False)
        for w in words:
            corpus_counts[w] += 1
            # Collect examples from low-confidence utterances (most likely to contain errors)
            if c < LOW_CONFIDENCE_THRESHOLD and len(corpus_examples[w]) < 20:
                corpus_examples[w].append(utt_text.strip())

    confusion = {}

    for target in seeds:
        variants         = {}   # variant_word -> frequency
        variant_examples = {}   # variant_word -> [example utterances]

        t_sdx = jellyfish.soundex(target)
        t_mph = jellyfish.metaphone(target)

        for corpus_word, cnt in corpus_counts.items():
            if corpus_word == target or cnt < MIN_WORD_FREQ:
                continue

            sdx_match = jellyfish.soundex(corpus_word) == t_sdx
            mph_match = jellyfish.metaphone(corpus_word) == t_mph
            jw_score  = jellyfish.jaro_winkler_similarity(corpus_word, target)
            # FIX: ratio-based edit distance — must be <= 30% of word length
            lev_r     = lev_ratio(corpus_word, target)

            is_variant = (
                sdx_match or
                mph_match or
                jw_score >= 0.82 or
                lev_r <= 0.30          # replaces the old fixed ≤2 / ≤3 thresholds
            )

            if is_variant:
                variants[corpus_word] = cnt
                # FIX: examples come from utterances containing the VARIANT word
                variant_examples[corpus_word] = corpus_examples.get(
                    corpus_word, []
                )[:MAX_EXAMPLES_PER_VARIANT]

        if variants:
            confusion[target] = {
                "variants":         variants,
                "variant_examples": variant_examples,
            }

    print(f"  Words with confusion variants found: {len(confusion):,}")
    return confusion


def format_confusion_report(confusion: dict, master: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten confusion map into a tidy DataFrame.
    Each row = one target word, with variants + labelled example utterances.
    Sorted by total variant frequency (how often the target is being confused),
    not raw variant count.
    """
    rows = []
    rank_lookup = {}
    if not master.empty:
        rank_lookup = {r["word"]: idx + 1
                       for idx, r in master.reset_index(drop=True).iterrows()}

    for word, data in confusion.items():
        variants_sorted    = sorted(data["variants"].items(), key=lambda x: -x[1])
        total_variant_freq = sum(c for _, c in variants_sorted)
        variants_str       = ", ".join(f"{v} ({c}x)" for v, c in variants_sorted[:10])

        # FIX: build examples labelled by which variant they came from
        example_parts = []
        for variant_word, _ in variants_sorted[:5]:
            exs = data["variant_examples"].get(variant_word, [])
            for ex in exs[:MAX_EXAMPLES_PER_VARIANT]:
                example_parts.append(f'[~{variant_word}] "{ex}"')
        examples_str = " | ".join(example_parts[:6])

        rows.append({
            "target_word":               word,
            "rank_in_master":            rank_lookup.get(word, "-"),
            "total_variant_frequency":   total_variant_freq,   # FIX: sort key
            "variant_count":             len(variants_sorted),
            "mistranscription_variants": variants_str,
            "example_utterances":        examples_str,
        })

    # FIX: sort by total variant frequency, not variant count
    return (pd.DataFrame(rows)
            .sort_values("total_variant_frequency", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────
#  MASTER REPORT — combine all 4 signals
# ─────────────────────────────────────────────
def build_master(s1, s2, s3, s4) -> pd.DataFrame:
    print(f"\n{Fore.CYAN}Combining signals into master report...")
    scores = defaultdict(lambda: {
        "word": "", "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0, "evidence": []
    })

    if not s1.empty:
        for _, r in s1.head(400).iterrows():
            w = r["word"]; scores[w]["word"] = w
            scores[w]["s1"] = min(float(r.get("signal_score", 1)), 10)
            scores[w]["evidence"].append(f"low-conf x{r.get('signal_score', '?'):.2f}")

    if not s2.empty:
        for _, r in s2.iterrows():
            w = r["word"]; scores[w]["word"] = w
            scores[w]["s2"] = 2.0
            hint = r.get("phonetically_similar_to", "")
            scores[w]["evidence"].append(f"OOV{'; ~' + hint if hint else ''}")

    if not s3.empty:
        for _, r in s3.iterrows():
            w = r["word"]; scores[w]["word"] = w
            scores[w]["s3"] = float(r.get("similarity_score", 0.8)) * 3
            scores[w]["evidence"].append(
                f"sounds like '{r.get('possible_intended_word', '')}' ({r.get('similarity_score', '')})")

    if not s4.empty:
        for _, r in s4.head(400).iterrows():
            w = r["word"]; scores[w]["word"] = w
            scores[w]["s4"] = float(r.get("anomaly_score", 0)) * 2
            scores[w]["evidence"].append(
                f"intent anomaly={r.get('anomaly_score', '?')} avg_conf={r.get('avg_confidence', '?')}")

    rows = []
    for word, d in scores.items():
        if not word or len(word) < 2:
            continue
        total = d["s1"] + d["s2"] + d["s3"] + d["s4"]
        fired = sum([d["s1"] > 0, d["s2"] > 0, d["s3"] > 0, d["s4"] > 0])
        rows.append({
            "word":            word,
            "composite_score": round(total, 3),
            "signals_fired":   fired,
            "sig_low_conf":    round(d["s1"], 3),
            "sig_oov":         round(d["s2"], 3),
            "sig_phonetic":    round(d["s3"], 3),
            "sig_intent_anom": round(d["s4"], 3),
            "evidence":        " | ".join(d["evidence"][:3]),
        })

    out = (pd.DataFrame(rows)
           .sort_values(["signals_fired", "composite_score"], ascending=[False, False])
           .reset_index(drop=True))
    out.index += 1
    out.index.name = "rank"
    return out


# ─────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────
def print_confusion_summary(confusion_df: pd.DataFrame, n: int = 20):
    # FIX: Style now properly imported
    print(f"\n{'='*70}")
    print(f" TOP {n} MISTRANSCRIPTION CONFUSION MAP")
    print(f"{'='*70}")
    for _, row in confusion_df.head(n).iterrows():
        print(
            f"\n  {Fore.YELLOW}{row['target_word'].upper()}{Style.RESET_ALL}  "
            f"(rank #{row['rank_in_master']}, "
            f"{row['variant_count']} variants, "
            f"{row['total_variant_frequency']} total occurrences)"
        )
        print(f"    Variants : {row['mistranscription_variants']}")
        if row["example_utterances"]:
            for ex in row["example_utterances"].split(" | "):
                print(f"    Example  : {ex}")
    print()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="STT Mistranscription Detector v2.1 — Voicebot STT Analytics")
    parser.add_argument("--input",  required=True,            help="CSV or Excel input file")
    parser.add_argument("--output", default="./stt_results",  help="Output folder")
    parser.add_argument("--config", default=None,             help="JSON file to override COLUMN_CONFIG")
    parser.add_argument("--top",    type=int, default=50,
                        help="Top N flagged words to also add to confusion map seeds")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = COLUMN_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    print(f"\n{'='*60}")
    print(f" STT Mistranscription Detector v2.1 — Voicebot Analytics")
    print(f"{'='*60}")

    setup_nltk()
    df      = load_data(args.input)
    col_map = map_columns(df, config)

    print(f"\n{Fore.CYAN}Column mapping:")
    for k, v in col_map.items():
        icon = Fore.GREEN + "✓" if v else Fore.YELLOW + "–"
        print(f"  {icon}{Style.RESET_ALL} {k}: {v or 'not available'}")

    # ── Run all 4 signals ────────────────────────────────────────────────────
    s1 = signal_low_confidence(df, col_map)
    s2 = signal_oov(df, col_map)
    s3 = signal_phonetic(df, col_map)
    s4 = signal_intent_anomaly(df, col_map)

    master = build_master(s1, s2, s3, s4)

    # ── Confusion map seeds ──────────────────────────────────────────────────
    # Seed from: (a) curated high-risk domain words + (b) top N from master report
    top_flagged = (list(master.head(args.top)["word"].dropna())
                   if not master.empty else [])
    all_seeds   = list(dict.fromkeys(list(CONFUSION_MAP_SEEDS) + top_flagged))

    confusion    = build_confusion_map(df, col_map, all_seeds)
    confusion_df = format_confusion_report(confusion, master)

    # ── Save outputs ─────────────────────────────────────────────────────────
    master.to_csv(out_dir / "master_mistranscription_candidates.csv")
    confusion_df.to_csv(out_dir / "confusion_map.csv", index=False)

    for fname, result in [
        ("signal_1_low_confidence_words.csv", s1),
        ("signal_2_oov_words.csv",            s2),
        ("signal_3_phonetic_collisions.csv",  s3),
        ("signal_4_intent_anomalies.csv",     s4),
    ]:
        if not result.empty:
            result.to_csv(out_dir / fname, index=False)

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" MASTER — TOP 20 SUSPECTED MISTRANSCRIBED WORDS")
    print(f"{'='*60}")
    if not master.empty:
        cols = [c for c in ["word", "composite_score", "signals_fired", "evidence"]
                if c in master.columns]
        print(master.head(20)[cols].to_string())

    print_confusion_summary(confusion_df, n=20)

    print(f"{Fore.GREEN}Results saved to: {out_dir.resolve()}/")
    print(f"  confusion_map.csv                      <- KEY OUTPUT")
    print(f"  master_mistranscription_candidates.csv <- ranked suspect words")
    print(f"  signal_1_low_confidence_words.csv")
    print(f"  signal_2_oov_words.csv")
    print(f"  signal_3_phonetic_collisions.csv")
    print(f"  signal_4_intent_anomalies.csv")
    print(f"\n{Fore.CYAN}Custom Dictionary Management — next steps:")
    print(f"  1. Open confusion_map.csv")
    print(f"  2. For each target_word, 'mistranscription_variants' are your boost candidates")
    print(f"  3. Check example_utterances to confirm — each is labelled [~variant] so you")
    print(f"     can see exactly which call contained the suspected error")
    print(f"  4. Add confirmed pairs to your STT platform's custom dictionary")
    print(f"     (e.g. Genesys: Admin > Speech & Text Analytics > Custom Dictionary)")
    print(f"  5. Re-run after dictionary updates to measure improvement\n")


if __name__ == "__main__":
    main()
