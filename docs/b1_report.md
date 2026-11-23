# B1: classifier report

Reproduce: `python ml/audit.py` then `python ml/train.py` (needs `data/bitext.csv`, see README).

## What I found in the data (before training)

| Finding | Numbers | What I did |
|---|---|---|
| Exact duplicate messages | 2,237 of 26,872 | Dropped |
| Duplicates after lower-casing, removing punctuation and removing `{{placeholders}}` | 3,164 | Dropped (23,708 rows left). No duplicate text carries two different labels, so nothing was lost to conflicts. |
| Template data: 24.8% of rows contain `{{Order Number}}`-style tokens, and most messages have a near-identical sibling | 7,946 near-duplicate groups (char-n-gram cosine ≥ 0.85); 6,139 are single rows, the largest has 245 | Split by group, never by row (below) |
| Class imbalance | ACCOUNT 5,986 vs CANCEL 950 | `class_weight="balanced"`; report macro-F1 as well as accuracy |
| Doubtful category design | `CANCEL` contains only `check_cancellation_fee`, while `cancel_order` is in `ORDER` and `delete_account` is in `ACCOUNT`. "When will purchase {{Order Number}} arrive" is `ORDER/track_order` but the same idea without an order number is `DELIVERY/delivery_period`. All 255 messages containing "ETA" are labelled `ORDER/track_order`. | Kept the labels (relabelling 27k rows by hand is outside 4 hours), but treat these confusions as label noise, not model failure, when explaining errors |
| Near-identical phrasings with different categories | 6 groups | Left as is |
| Style | Short, clean, mostly one sentence; some mild typos and slang ("damn", "oredr") | Added character n-grams so typos still match; wrote 20 messier messages to check |

## Why the split does not leak

1. Exact and normalised duplicates were removed first, so no identical text can be in both train and test (asserted in `train.py`).
2. Near-duplicates are grouped (connected components of the "cosine ≥ 0.85" graph) and `GroupShuffleSplit` puts each whole group on one side. `train.py` asserts the train and test groups do not overlap.
3. Final model is fit on everything only **after** the numbers are measured; the reported scores come from models that never saw their test rows.
4. Caveat: 0.85 catches template siblings (a dropped word scores ~0.99) but **not** typo variants (0.72 to 0.82). I checked this matters little: at grouping thresholds 0.9, 0.85, 0.8 and 0.7 the held-out accuracy stayed at 99.9 to 100%.

## Scores

| Evaluation | Accuracy | Macro-F1 |
|---|---|---|
| Random split on raw rows (deliberately leaky baseline) | 99.98% | 0.9999 |
| Grouped split, 18,435 train / 5,273 test | 99.98% (1 error) | 0.9999 |
| Grouped 5-fold CV over all 23,708 rows | 99.98% (5 errors) | 0.9998 |
| **20 hand-written messy messages** (`ml/messy_20.csv`) | **17/20 = 85%** (evaluation model); 18/20 for the shipped model refit on all data | n/a |

**How to read this:** the in-distribution score is saturated. The leaky and the clean split give the same number, which means the data is easy (templated, keyword-separable), not that the split leaks. That score tells you little about real tickets. The messy-20 score is the more meaningful number, but n = 20, so one message is 5 points; the evaluation and shipped models differ by one message. The 20 messages and their labels were written before any model output was seen.

## Five wrong predictions, explained (with the probe that supports each)

1. **"my credit card got declined but i have money on it"** (PAYMENT, predicted REFUND at 0.50). The training data says "payment failed/issue", never "declined". Dropping "money" gives ACCOUNT 0.38 instead; rewording to "payment failed" gives PAYMENT 0.95. The word "money" drags it toward REFUND ("get my money back" phrases). Vocabulary gap.
2. **"is there a fee if i cancel my order after it shipped"** (CANCEL, predicted ORDER at 0.55). "cancel my order" is overwhelmingly ORDER/cancel_order in this data. Removing "my order" ("fee if i cancel after it shipped") flips it to CANCEL 0.45. Also a label-design issue: CANCEL holds only cancellation fees, and ORDER is defensible for this message.
3. **"can i change the size on my order before it ships"** (ORDER, predicted ACCOUNT in the evaluation model; ORDER 0.30 vs ACCOUNT 0.27 in the shipped one). "can i change my order" is ORDER 0.85, but adding "the size on" drops it to 0.49. Template "change … on my …" phrasing resembles account edits. Low confidence either way.
4. **"need help to place an oredr"** (ORDER, predicted ACCOUNT in one CV fold). With the typo fixed the model gives ORDER 0.98; with the typo it is 0.45 even in a model that never saw this message's group. The typo destroys the key word and char n-grams only partly recover it.
5. **"i have to see the damn ETA of my delivery helpme"** (labelled ORDER/track_order, predicted DELIVERY at 0.76). The model is probably right and the label doubtful: all 255 "ETA" rows are ORDER/track_order, but this one asks about delivery time.

## What the app does with this

The app stores the model's confidence next to each category so agents can see weak predictions (the misses above are all low-confidence). It does not auto-route on category.
