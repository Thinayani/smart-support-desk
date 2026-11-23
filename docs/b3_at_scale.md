# B3: measuring reply quality at 10,000 tickets a day

Nobody can read 10,000 drafts a day, so measure in layers, cheapest first.

**1. Checks on every ticket (free, automatic, run inline).** What exists today: every ticket stores its draft source (`no-match`, `faq-template`, `llm`, `fallback`) and the FAQ ids used, so the first item is a SQL query. The rest would need counters added (not built):
- Share of drafts by source. A jump in `fallback` means the model is producing output the validator rejects; a jump in `no-match` means a new topic or a retrieval regression.
- Validator rejections by reason (unknown FAQ id, commitment phrase, too much wording not in the FAQ). Not recorded yet; `validate_llm_output` returns only `None`.
- Retrieval score distribution. Drift in it is the earliest sign something changed.
- Latency and LLM error rate.

**2. Human outcome signals (cheap, real).** The team page is where agents would use the draft, so it should log (not built):
- Sent as is / edited a little / edited a lot / discarded. Edit distance between draft and final reply is the best proxy for quality.
- Ticket reopened or customer replied "that didn't answer my question" within 48 h.
Track these per FAQ entry and per category: a bad FAQ entry or a weak category shows up as a hot spot.

**3. A small labelled sample (the ground truth).** Each day sample about 100 tickets, stratified so that rare categories, low retrieval scores, `no-match` and `fallback` drafts are over-represented. Two reviewers grade good/bad against the rubric used in `ml/eval_tickets.json` (right FAQ entry? covers every question? invents nothing? promises nothing?). 100 a day gives about ±10 points per day and about ±3 points per week on the overall good rate; keep disagreements to calibrate reviewers.

**4. LLM-as-judge to extend the sample, not replace it.** Run a judge model over a larger slice (say 2,000 a day) with the same rubric, and report its agreement with the human sample each week. Only trust it on the dimensions where agreement is high (e.g. "did it invent anything"). Pin the judge model version.

**5. A fixed regression set run on every change.** The 10 tickets here grow into a few hundred, with every failure found in production added. Run it on each prompt, model, FAQ or threshold change; block releases that lower the good rate or let any injection case through.

**Alerts:** `no-match` share, `fallback` share, edit-distance median, and good-rate on the daily sample, each against a rolling baseline.

**Safety is measured separately from quality:** any draft that promises a refund, reveals the system prompt or follows an instruction from the ticket counts as an incident, not as a "bad" grade. Replay known injection phrasings daily.
