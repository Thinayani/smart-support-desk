# NOTES

## Design choices and trade-offs
- **Small FastAPI app, SQLite, plain HTML pages.** Easy to run and read. Trade-off: one shared team login from env vars, no rate limit on the public form.
- **Classifier: TF-IDF (word + char n-grams) + logistic regression.** Char n-grams help with typos, and the data is easy. The app stores and shows the model's confidence; it never routes on category alone.
- **Honest split:** dedupe, then group near-duplicates and keep each group on one side (`docs/b1_report.md`). The leaky and the clean split both score 99.98%, so Bitext is too easy to say much about real tickets. My 20 messy messages (17/20) are more telling, but n = 20.
- **Replies are grounded by construction.** Retrieval runs first and is deterministic. With no matching FAQ entry the model is never called and the draft says it has no answer. If it is called, its JSON must cite retrieved entries, promise nothing, and stay close to the FAQ wording, or a fixed FAQ-based draft is used. The ticket is quoted data, never instructions.
- **Drafts run in a background task**, so the form never waits on an LLM.
- **Eval: 6/10 good** on my hard tickets (`docs/b3_results.md`). Failures: a double charge got the cancellation-fee answer, "delivered but missing" got generic tracking info, one of two questions was answered, a typo-heavy ticket was missed. Lexical retrieval is the weak part.

## Not verified
- The Gemini and Ollama clients ran only against a faked network (no key or model in my sandbox), so B3 used the FAQ-only path. Free-tier Gemini content may be used by Google to improve its products; only ticket text and FAQ entries are sent.
- The Docker files were not built (no Docker daemon here). A clean-venv install with the same files works.

## Two places the AI tool gave wrong output
1. **Grouping code skipped the first neighbour, assuming it was the row itself.** With exact ties it is not, so some duplicate pairs were never linked. A unit test with two identical texts failed. I fixed it and retrained; scores did not change.
2. **Wrong first thresholds.** Grouping at 0.80 chained 2,810 rows into one group and left CANCEL with 22 test rows; I saw it by printing class counts and moved to 0.85. A retrieval threshold of 0.30 said "no match" to clear questions like "where is my order"; a dev set showed it, and I set 0.20.

## With one more week
1. Run B3 with a real local LLM and compare it to the deterministic path.
2. Better retrieval (embeddings plus reranker) and answers that combine two FAQ entries.
3. Merge or relabel confusing categories (CANCEL/ORDER, ORDER/DELIVERY); build a larger, messier labelled test set.
4. Log agent actions on drafts (sent, edited, discarded) and validator rejection reasons.
5. Real auth with hashed passwords, rate limiting, Postgres.
