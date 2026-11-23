# Smart Support Desk

A small support-desk web app with an ML ticket classifier and grounded AI reply drafts.

- **Customer page** `/`: send a message (name, email, message).
- **Team page** `/team`: login, list tickets with category / priority / status, filter by category, open a ticket, change its status, see the suggested reply and the FAQ entries it used.
- **API** under `/api` (interactive docs at `/docs`). SQLite database.

## Run it

**With Docker (one command):**

```bash
docker compose up --build
```

**Without Docker:**

```bash
make install   # pip install -r requirements-dev.txt
make run       # http://localhost:8000
```

Open http://localhost:8000 and http://localhost:8000/team. Demo login: `agent` / `changeme`.
Set real values with a `.env` file (copy `.env.example`); `.env` is git-ignored. **No API keys are needed or stored.**

The trained model is committed (`ml/artifacts/classifier.joblib`, 1.8 MB), so nothing has to be downloaded to run the app.

## Tests

```bash
make test      # 48 tests: API, model + split hygiene, reply logic (incl. prompt injection), Gemini client (network faked)
```

## Reproduce the ML work

```bash
make data      # downloads Bitext to data/bitext.csv (or save the CSV there yourself)
make audit     # what is wrong with the data
make train     # leaky vs grouped split, 5-fold CV, 20 messy messages; writes the model
make eval      # B3: reply drafts on 10 hard tickets -> docs/b3_results.md
```

Reports: `docs/b1_report.md` (data problems, split, scores, 5 explained errors), `docs/b3_results.md` (10-ticket evaluation), `docs/b3_at_scale.md` (measuring quality at 10,000 tickets a day). Design notes: `NOTES.md`.

## Reply drafts and the LLM

By default `LLM_PROVIDER=none`: drafts are built directly from the best-matching FAQ entry (`faq.json`, 15 entries), with no model call. To let an LLM phrase them, pick one:

**Gemini (hosted; a free key works).** Create a key at https://aistudio.google.com/apikey, then in `.env`:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
```

Free-tier model names change, so check https://ai.google.dev/gemini-api/docs/pricing if you get "fallback" drafts. Note that Google's pricing page says free-tier content may be used to improve its products. Only the ticket text and FAQ entries are sent, never the customer's name or email, but use a paid key for real customer data. The key is read from the environment and must never be committed.

**Ollama (local).**

```bash
docker compose --profile llm up -d
docker compose exec ollama ollama pull llama3.2:3b
# in .env: LLM_PROVIDER=ollama
```

After changing `.env`, restart (`docker compose down && docker compose up --build`). A ticket's draft tag shows which path produced it: `llm`, `faq-template` (no LLM configured) or `fallback` (an LLM was configured but failed or was rejected).

Either way the same rules apply: no matching FAQ entry means an honest "I could not find an answer" draft and no model call; model output is validated against the retrieved entries and replaced by the FAQ-based draft if it fails. See `app/replies.py`.

## Layout

```
app/        FastAPI app: main (routes), db, auth, schemas, classifier, replies
static/     customer and team pages (all user text rendered with textContent)
ml/         audit.py, train.py, eval_replies.py, messy_20.csv, eval_tickets.json, artifacts/
faq.json    FAQ used for reply drafts
tests/      pytest suite
docs/       reports
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ADMIN_USER` / `ADMIN_PASSWORD` | `agent` / `changeme` | Team login. Change both outside a demo. |
| `SECRET_KEY` | random per start | Signs login tokens |
| `DB_PATH` | `data/support.db` | SQLite file |
| `LLM_PROVIDER` | `none` | `none`, `gemini` or `ollama` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | empty, `gemini-2.5-flash` | Hosted LLM; key from the environment only |
| `OLLAMA_URL`, `OLLAMA_MODEL` | `http://localhost:11434`, `llama3.2:3b` | Local LLM |
| `FAQ_MIN_SCORE` | `0.20` | Minimum retrieval score before any draft is attempted |

## Known limits

Single shared login, no rate limiting on the public form, SQLite, priority is a transparent keyword rule (not a model), English only. The Docker files have not been built in the environment this was written in; the non-Docker path was tested from a clean virtualenv. The LLM paths (Gemini, Ollama) are covered by tests with a faked network only; they have not been run against a real model.
