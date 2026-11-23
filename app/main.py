"""REST API + static pages for the Smart Support Desk."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import auth, classifier, config, db, replies
from .schemas import STATUSES, LoginIn, StatusIn, TicketIn

log = logging.getLogger("support-desk")
STATIC = Path(__file__).resolve().parent.parent / "static"
CATEGORIES = ["ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE", "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    classifier._model()  # fail fast if the model file is missing
    if config.LLM_PROVIDER == "gemini" and not config.GEMINI_API_KEY:
        log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty: every draft will use the FAQ fallback.")
    log.info("reply drafts: provider=%s", config.LLM_PROVIDER)
    yield


app = FastAPI(title="Smart Support Desk", lifespan=lifespan)


def error(status: int, code: str, message: str, fields: Optional[list] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "fields": fields or []}})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    fields = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"].removeprefix("Value error, ")}
        for e in exc.errors()
    ]
    return error(422, "validation_error", "Some fields are invalid.", fields)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return error(exc.status_code, "http_error", str(exc.detail))


def _make_draft(ticket_id: int, message: str, category: str) -> None:
    try:
        d = replies.draft_reply(message, category)
        db.save_draft(ticket_id, d.text, d.faq_ids, d.grounded, d.source, ok=True)
    except Exception:  # a failed draft must never lose the ticket
        log.exception("draft generation failed for ticket %s", ticket_id)
        db.save_draft(ticket_id, None, [], False, "error", ok=False)


@app.post("/api/tickets", status_code=201)
def create_ticket(payload: TicketIn, background: BackgroundTasks):
    category, conf = classifier.classify(payload.message)
    prio = classifier.priority(category, payload.message)
    ticket_id = db.create_ticket(payload.name, payload.email, payload.message, category, conf, prio)
    background.add_task(_make_draft, ticket_id, payload.message, category)
    # Customers get a receipt only; category, priority and draft are for the team.
    return {"id": ticket_id, "message": "Thanks, we have received your message."}


@app.post("/api/login")
def login(payload: LoginIn):
    if not auth.check_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return {"token": auth.make_token(payload.username)}


@app.get("/api/meta")
def meta(_: str = auth.TeamUser):
    return {"categories": CATEGORIES, "statuses": list(STATUSES)}


@app.get("/api/tickets")
def list_tickets(
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    _: str = auth.TeamUser,
):
    if category and category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"Unknown category. Use one of: {', '.join(CATEGORIES)}.")
    if status and status not in STATUSES:
        raise HTTPException(status_code=422, detail=f"Unknown status. Use one of: {', '.join(STATUSES)}.")
    return db.list_tickets(category, status)


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: int, _: str = auth.TeamUser):
    t = db.get_ticket(ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    faq = {e["id"]: e for e in replies.load_faq()}
    t["draft_faq_entries"] = [
        {"id": i, "question": faq[i]["question"], "answer": faq[i]["answer"]} for i in t["draft_faq_ids"] if i in faq
    ]
    return t


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: int, payload: StatusIn, _: str = auth.TeamUser):
    if not db.update_status(ticket_id, payload.status):
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return db.get_ticket(ticket_id)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
def customer_page():
    return FileResponse(STATIC / "index.html")


@app.get("/team", include_in_schema=False)
def team_page():
    return FileResponse(STATIC / "team.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
