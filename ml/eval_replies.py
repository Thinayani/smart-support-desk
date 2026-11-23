"""B3: run reply drafts on the 10 hand-written hard tickets and grade them against expectations
that were written BEFORE the system was run (see ml/eval_tickets.json).

Run:  python ml/eval_replies.py            (uses LLM_PROVIDER from the environment; default: none)
      LLM_PROVIDER=gemini GEMINI_API_KEY=... python ml/eval_replies.py
      LLM_PROVIDER=ollama python ml/eval_replies.py
Writes docs/b3_results.md
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import classifier, config, replies  # noqa: E402


def grade(ticket: dict, draft: replies.Draft) -> tuple[bool, str]:
    for pat in ticket.get("forbidden", []):
        if re.search(pat, draft.text, re.I):
            return False, f"draft contains forbidden content /{pat}/"
    no_match = draft.source == "no-match"
    expect = ticket["expect"]
    if expect == "no_match":
        if no_match:
            return True, "correctly said it has no answer"
        return False, f"answered from {draft.faq_ids} although no FAQ entry covers this"
    if expect == "answer_or_no_match":
        if no_match:
            return True, "honest no-match (acceptable)"
        allowed = set(ticket.get("allowed_faq", []))
        if set(draft.faq_ids) <= allowed:
            return True, f"showed {draft.faq_ids}, which is allowed"
        return False, f"cited {draft.faq_ids}, outside {sorted(allowed)}"
    required = set(ticket["required_faq"])
    if no_match:
        return False, "said no match although the FAQ covers it"
    missing = required - set(draft.faq_ids)
    if missing:
        return False, f"missing {sorted(missing)}; cited {draft.faq_ids}"
    return True, f"cited {draft.faq_ids}"


def main() -> None:
    tickets = json.loads((ROOT / "ml" / "eval_tickets.json").read_text(encoding="utf-8"))
    provider = config.LLM_PROVIDER
    rows, good = [], 0
    for t in tickets:
        category, conf = classifier.classify(t["text"])
        d = replies.draft_reply(t["text"], category)
        ok, reason = grade(t, d)
        good += ok
        rows.append((t, category, d, ok, reason))
        print(f"#{t['id']:>2} {'GOOD' if ok else 'BAD '} [{d.source:12}] {category:12} {d.faq_ids}  {reason}")

    lines = [
        "# B3 results: reply drafts on 10 hard tickets",
        "",
        f"LLM provider for this run: `{provider}` ("
        + (
            "deterministic FAQ-only drafts, no model call" if provider == "none"
            else f"model `{config.GEMINI_MODEL if provider == 'gemini' else config.OLLAMA_MODEL}`"
        )
        + f"). Score: **{good}/{len(rows)} good**. Expectations were written before running (`ml/eval_tickets.json`).",
        "",
        "| # | Ticket | Predicted category | Draft source | FAQ used | Verdict | Reason |",
        "|---|--------|--------------------|--------------|----------|---------|--------|",
    ]
    for t, category, d, ok, reason in rows:
        text = t["text"].replace("|", "\\|")
        lines.append(f"| {t['id']} | {text} | {category} | {d.source} | {', '.join(d.faq_ids) or '-'} | "
                     f"{'good' if ok else 'BAD'} | {reason} |")
    lines += ["", "## Draft text per ticket", ""]
    for t, category, d, ok, reason in rows:
        lines += [f"**#{t['id']}** ({'good' if ok else 'BAD'}): _{t['why']}_", "", "```", d.text, "```", ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "b3_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{good}/{len(rows)} good -> docs/b3_results.md")


if __name__ == "__main__":
    main()
