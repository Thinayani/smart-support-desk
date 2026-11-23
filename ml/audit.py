"""Look at the Bitext data before training. Prints the problems the README/NOTES refer to.

Run:  python ml/audit.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dedupe, load_raw, near_dup_groups  # noqa: E402


def main() -> None:
    raw = load_raw()
    print(f"rows: {len(raw)}  columns: {list(raw.columns[:5])}")
    print("\nclass balance (raw):")
    print(raw["category"].value_counts().to_string())

    ph = raw["instruction"].str.contains(r"\{\{.*?\}\}")
    print(f"\nrows with {{{{placeholder}}}} tokens: {ph.sum()} ({ph.mean():.1%})")

    print(f"\nexact duplicate instructions: {raw['instruction'].duplicated().sum()}")
    print(f"duplicates after normalising case/punctuation/placeholders: {raw['norm'].duplicated().sum()}")
    conflict = (raw.groupby("norm")["category"].nunique() > 1).sum()
    print(f"duplicate texts carrying different labels: {conflict}")

    df = dedupe(raw)
    df["group"] = near_dup_groups(df["norm"])
    sizes = df.groupby("group").size()
    print(f"\nafter dedupe: {len(df)} rows in {len(sizes)} near-duplicate groups "
          f"({(sizes == 1).sum()} singletons, largest group {sizes.max()})")
    mixed = (df.groupby("group")["category"].nunique() > 1).sum()
    print(f"near-duplicate groups that span more than one category (label doubt): {mixed}")

    print("\nintents per category (CANCEL has a single intent; cancel_order lives in ORDER):")
    print(df.groupby("category")["intent"].unique().apply(lambda a: ", ".join(sorted(a))).to_string())

    print("\nsample of near-identical phrasings with different categories:")
    mixed_df = df[df.groupby("group")["category"].transform("nunique") > 1]
    for n, (_, g) in enumerate(mixed_df.groupby("group")):
        if n >= 3:
            break
        print(g[["instruction", "category", "intent"]].head(6).to_string(index=False))
    print("\n(Also: 'when will purchase {{Order Number}} arrive' is labelled ORDER/track_order while the same "
          "idea without an order number is DELIVERY/delivery_period.)")


if __name__ == "__main__":
    main()
