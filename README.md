# STT Analytics Toolkit

Tooling for diagnosing and improving Speech-to-Text (STT) quality in production voicebot systems. Built from real-world conversational AI work on enterprise-scale call centre deployments.

Two focused tools — one for finding **what the STT model mishears**, one for analysing **where and why no-matches occur**.

---

## Background

Built solo between February and mid-March 2026 to address a gap in observability for two national telco voicebot deployments operating at enterprise call centre scale.

The input dataset covered **three months of production NLP call data** across both brands — tens of thousands of sessions — giving the analysis enough volume to surface statistically meaningful mistranscription patterns rather than one-off noise.

The confusion map output (`confusion_map.csv`) was used directly to action changes in **Genesys Cloud's custom dictionary management**, boosting phonetically ambiguous domain terms that the STT model was consistently mishearing. This moved the work from insight to a concrete system change — reducing no-match rates driven by transcription errors rather than genuine user intent failures.

The no-match metrics tool was developed in parallel to quantify the *shape* of failures: specifically, how many no-match events were caused by short, ambiguous utterances versus longer ones where the STT model had more signal to work with.

---

## Tools

### 1. STT Mistranscription Detector (`stt_mistranscription_detector/`)

Identifies likely mistranscribed words in voicebot utterance data by running four independent signals and combining them into a ranked master report.

**Signals:**
| Signal | What it catches |
|---|---|
| Low confidence | Words appearing frequently in low-confidence utterances |
| Out-of-vocabulary (OOV) | Words not in standard English but phonetically close to domain terms |
| Phonetic collision | Words that sound like known domain vocab (Soundex + Levenshtein) |
| Intent anomaly | Words that appear consistently in misclassified or low-confidence intents |

**Output:** A confusion map showing what each domain term is being misheard as, with real example utterances — ready to feed directly into your STT platform's custom dictionary.

```bash
python stt_mistranscription_detector/stt_mistranscription_engine.py \
  --input data/utterances.csv \
  --output results/
```

**Output files:**
```
results/
├── master_mistranscription_candidates.csv   ← ranked flagged words
├── confusion_map.csv                        ← key output: target → variants
├── signal_1_low_confidence_words.csv
├── signal_2_oov_words.csv
├── signal_3_phonetic_collisions.csv
└── signal_4_intent_anomalies.csv
```

---

### 2. No-Match Metrics (`no_match_metrics.py`)

Analyses `NO_MATCH` events in conversational AI session logs to understand failure patterns — specifically whether short utterances (≤2 words) are driving no-match rates, and whether users naturally self-correct on their next turn.

```bash
python no_match_metrics.py data/session_logs.csv
```

**Example output:**
```
NO_MATCH Analysis  |  March, Days 10–13
────────────────────────────────────────
Total NO_MATCH rows:         1,243
≤ 2 words (short):            847  (68.1%)
> 2 words (long):             396  (31.9%)

Follow-up improvement (after ≤2-word NO_MATCH):
  ✅ Next turn had MORE words:   612  (72.3%)
  ❌ Did NOT improve:            198  (23.4%)
  ⚠️  No next turn found:         37   (4.4%)
```

Useful for deciding whether to prompt rephrasing, add fallback handlers, or retrain on short-utterance edge cases.

---

## Setup

```bash
git clone https://github.com/your-username/stt-analytics-toolkit.git
cd stt-analytics-toolkit

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your column names and thresholds
```

---

## Configuration

Edit `stt_mistranscription_detector/config.py` to match your platform's data export:

```python
COLUMN_CONFIG = {
    "utterance_col":  "Utterance",
    "intent_col":     "Intent",
    "confidence_col": "Intent Confidence",
    "session_col":    "Session ID",
}
LOW_CONFIDENCE_THRESHOLD = 0.60  # adjust for your platform's scoring
```

The domain vocabulary (`DOMAIN_VOCAB`) is seeded for a telco environment — replace or extend with your own product names, plan names, and terminology.

---

## Input Data Format

Both tools expect a CSV or Excel export from your voicebot/conversation platform (Genesys Cloud, Dialogflow CX, CCAI, etc.) with columns for utterance text, intent, confidence score, and session ID. Column names are configurable — no reformatting required.

**Real data should never be committed.** Add your data files to a local `data/` folder (already in `.gitignore`).

---

## Use Cases

- **STT custom dictionary tuning** — use the confusion map to identify which domain terms to boost in your ASR model
- **NLU fallback analysis** — understand the shape of your no-match traffic before writing new training phrases
- **Pre/post deployment comparison** — run before and after dictionary updates to measure STT improvement
- **Voicebot QA** — surface systematic transcription errors that are invisible in aggregate metrics

---

## Stack

Python · pandas · NLTK · jellyfish (phonetic similarity) · colorama · openpyxl
