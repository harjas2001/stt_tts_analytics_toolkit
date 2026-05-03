import pandas as pd
import sys

def word_count(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0
    return len(str(text).strip().split())

def analyse(filepath):
    df = pd.read_csv(filepath)

    # ── 1. Filter: March, Days 10–13, NO_MATCH ──────────────────────────────
    filtered = df[
        (df["Month"].str.strip().str.lower() == "march") &
        (df["Day"].astype(int).isin([10, 11, 12, 13])) &
        (df["match_type"].str.strip() == "NO_MATCH")
    ].copy()

    # ── 2. Total row count ───────────────────────────────────────────────────
    total_rows = len(filtered)

    # ── 3. Word count split on user_utterance ───────────────────────────────
    filtered["word_count"] = filtered["user_utterance"].apply(word_count)
    short_utterances = (filtered["word_count"] <= 2).sum()
    long_utterances  = (filtered["word_count"] > 2).sum()

    # ── 4. Follow-up improvement check ──────────────────────────────────────
    # Base pool: same date filter (no match_type restriction) so we can look
    # at the very next turn regardless of its match type.
    base = df[
        (df["Month"].str.strip().str.lower() == "march") &
        (df["Day"].astype(int).isin([10, 11, 12, 13]))
    ].copy()
    base["word_count"] = base["user_utterance"].apply(word_count)
    base = base.sort_values(["session_id", "turn_position"])

    # Rows that were a NO_MATCH with ≤2 words (the trigger rows)
    trigger_rows = base[
        (base["match_type"].str.strip() == "NO_MATCH") &
        (base["word_count"] <= 2)
    ]

    improved   = 0
    not_improved = 0
    no_next_turn = 0

    for _, row in trigger_rows.iterrows():
        session = base[base["session_id"] == row["session_id"]]
        next_turns = session[session["turn_position"] == row["turn_position"] + 1]

        if next_turns.empty:
            no_next_turn += 1
        else:
            next_wc = next_turns.iloc[0]["word_count"]
            if next_wc > row["word_count"]:
                improved += 1
            else:
                not_improved += 1

    total_trigger = len(trigger_rows)

    # ── Print results ────────────────────────────────────────────────────────
    print("=" * 55)
    print("  NO_MATCH Analysis  |  March, Days 10–13")
    print("=" * 55)

    print(f"\n📊 Total NO_MATCH rows (filtered):  {total_rows}")

    print(f"\n📝 Utterance word-count split:")
    print(f"   ≤ 2 words  :  {short_utterances}  ({short_utterances/total_rows*100:.1f}%)")
    print(f"   > 2 words  :  {long_utterances}  ({long_utterances/total_rows*100:.1f}%)")

    print(f"\n🔁 Follow-up improvement (after ≤2-word NO_MATCH):")
    print(f"   Trigger rows (≤2-word NO_MATCH):  {total_trigger}")
    print(f"   ✅ Next turn had MORE words     :  {improved}  ({improved/total_trigger*100:.1f}% of triggers)" if total_trigger else "   No trigger rows found.")
    if total_trigger:
        print(f"   ❌ Next turn did NOT improve    :  {not_improved}  ({not_improved/total_trigger*100:.1f}% of triggers)")
        print(f"   ⚠️  No next turn found          :  {no_next_turn}  ({no_next_turn/total_trigger*100:.1f}% of triggers)")

    print("=" * 55)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_no_match.py <path_to_csv>")
        sys.exit(1)
    analyse(sys.argv[1])
