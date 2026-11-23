"""Train the ticket category classifier and write an honest report.

Run:  python ml/train.py
Outputs: ml/artifacts/classifier.joblib, ml/artifacts/report.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_PATH, ROOT, dedupe, load_raw, near_dup_groups, normalize  # noqa: E402

SEED = 42


def make_pipeline() -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ]
    )
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", feats), ("clf", clf)])


def fit_eval(train_x, train_y, test_x, test_y):
    pipe = make_pipeline().fit(train_x, train_y)
    pred = pipe.predict(test_x)
    return pipe, pred, accuracy_score(test_y, pred), f1_score(test_y, pred, average="macro")


def main() -> None:
    raw = load_raw()
    df = dedupe(raw)
    df["group"] = near_dup_groups(df["norm"])

    # 1) Deliberately leaky baseline: random split on RAW rows (duplicates and near-duplicates cross the split).
    tr, te = train_test_split(raw, test_size=0.2, random_state=SEED, stratify=raw["category"])
    _, _, leaky_acc, leaky_f1 = fit_eval(tr["instruction"], tr["category"], te["instruction"], te["category"])

    # 2) Honest split: dedupe, then keep each near-duplicate group wholly on one side.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, df["category"], groups=df["group"]))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    overlap_groups = set(tr["group"]) & set(te["group"])
    overlap_norm = set(tr["norm"]) & set(te["norm"])
    assert not overlap_groups and not overlap_norm, "split leaks"

    pipe, pred, acc, f1 = fit_eval(tr["instruction"], tr["category"], te["instruction"], te["category"])
    print(f"leaky random split : acc={leaky_acc:.4f} macro-F1={leaky_f1:.4f}")
    print(f"grouped split      : acc={acc:.4f} macro-F1={f1:.4f}  (train={len(tr)}, test={len(te)})")
    print(classification_report(te["category"], pred, digits=3))
    test_errors = te.assign(pred=pred)[pred != te["category"].values][["instruction", "category", "intent", "pred"]]
    print("held-out test errors:\n", test_errors.to_string(index=False))

    # 2b) Grouped 5-fold CV over everything: a steadier estimate and a bigger pool of real errors.
    oof = np.empty(len(df), dtype=object)
    for f_tr, f_te in GroupKFold(n_splits=5).split(df, df["category"], groups=df["group"]):
        oof[f_te] = make_pipeline().fit(df["instruction"].iloc[f_tr], df["category"].iloc[f_tr]).predict(
            df["instruction"].iloc[f_te]
        )
    cv_acc = accuracy_score(df["category"], oof)
    cv_f1 = f1_score(df["category"], oof, average="macro")
    cv_errors = df.assign(pred=oof)[oof != df["category"].values][["instruction", "category", "intent", "pred"]]
    print(f"grouped 5-fold CV  : acc={cv_acc:.4f} macro-F1={cv_f1:.4f}  errors={len(cv_errors)}/{len(df)}")
    print(cv_errors.head(12).to_string(index=False))

    # 3) The 20 hand-written messy messages (written before looking at any model output).
    messy = pd.read_csv(ROOT / "ml" / "messy_20.csv")
    messy["pred"] = pipe.predict(messy["text"])
    messy_acc = accuracy_score(messy["category"], messy["pred"])
    wrong = messy[messy["pred"] != messy["category"]]
    print(f"\nmessy-20 accuracy: {messy_acc:.2f} ({len(messy) - len(wrong)}/{len(messy)})")
    print(wrong.to_string(index=False))

    # 4) Final model for the app: refit on ALL deduped data (test score above came from the held-out fit).
    final = make_pipeline().fit(df["instruction"], df["category"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    report = {
        "rows_raw": len(raw),
        "rows_after_dedupe": len(df),
        "near_dup_groups": int(df["group"].nunique()),
        "leaky_random_split": {"accuracy": leaky_acc, "macro_f1": leaky_f1},
        "grouped_split": {"accuracy": acc, "macro_f1": f1, "train": len(tr), "test": len(te)},
        "test_errors": test_errors.to_dict("records"),
        "grouped_5fold_cv": {"accuracy": cv_acc, "macro_f1": cv_f1, "errors": int(len(cv_errors))},
        "cv_errors_sample": cv_errors.head(20).to_dict("records"),
        "messy_20_accuracy": messy_acc,
        "messy_20_wrong": wrong.to_dict("records"),
    }
    (MODEL_PATH.parent / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
