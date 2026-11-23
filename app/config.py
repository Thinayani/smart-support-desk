"""Settings come from environment variables. No secrets live in the repo."""
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = os.environ.get("DB_PATH", str(ROOT / "data" / "support.db"))
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "ml" / "artifacts" / "classifier.joblib"))
FAQ_PATH = os.environ.get("FAQ_PATH", str(ROOT / "faq.json"))

# Team login. Demo defaults so `docker compose up` works; override both in any real deployment.
ADMIN_USER = os.environ.get("ADMIN_USER", "agent")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

# Signing key for login tokens. A random per-process key is used if unset (tokens die on restart).
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
TOKEN_TTL_SECONDS = int(os.environ.get("TOKEN_TTL_SECONDS", "28800"))

# LLM for reply drafts: "gemini" (hosted, free tier available), "ollama" (local) or "none"
# (deterministic FAQ-only drafts, no model call).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none").lower()

# Gemini: the key is read from the environment only and is never stored in the repo or logged.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Free-tier model names change over time. Check https://ai.google.dev/gemini-api/docs/pricing and override if needed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))

# A ticket must match an FAQ entry at least this well before any draft is attempted.
# 0.20 chosen on a 24-query dev set: zero answers to unanswerable questions, at the cost of a few
# missed matches. A missed match yields an honest "I don't know" for the agent; a wrong answer is worse.
FAQ_MIN_SCORE = float(os.environ.get("FAQ_MIN_SCORE", "0.20"))
