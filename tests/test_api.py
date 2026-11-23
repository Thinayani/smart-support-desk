import time

from app import auth


def test_customer_can_submit_and_gets_only_a_receipt(client):
    r = client.post("/api/tickets", json={"name": "Ann", "email": "ann@example.com", "message": "I forgot my password"})
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"id", "message"}  # no category / priority / draft leaked to the customer


def test_validation_errors_are_clear_and_per_field(client):
    r = client.post("/api/tickets", json={"name": "  ", "email": "not-an-email", "message": "hi"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    fields = {f["field"] for f in err["fields"]}
    assert fields == {"name", "email", "message"}


def test_missing_body_and_oversized_message_rejected(client):
    assert client.post("/api/tickets", json={}).status_code == 422
    r = client.post("/api/tickets", json={"name": "A", "email": "a@b.co", "message": "x" * 2001})
    assert r.status_code == 422


def test_team_endpoints_require_login(client):
    assert client.get("/api/tickets").status_code == 401
    assert client.get("/api/tickets/1").status_code == 401
    assert client.patch("/api/tickets/1", json={"status": "closed"}).status_code == 401
    bad = client.get("/api/tickets", headers={"Authorization": "Bearer garbage"})
    assert bad.status_code == 401


def test_wrong_password_rejected(client):
    r = client.post("/api/login", json={"username": "agent", "password": "nope"})
    assert r.status_code == 401
    assert r.json()["error"]["message"]


def test_token_expiry_and_tampering():
    token = auth.make_token("agent", now=time.time() - 10**6)
    assert auth.verify_token(token) is None  # expired
    good = auth.make_token("agent")
    assert auth.verify_token(good) == "agent"
    assert auth.verify_token(good[:-4] + "AAAA") is None  # tampered


def test_ticket_is_classified_and_listed(client, auth_header, make_ticket):
    tid = make_ticket("how do i delete my account and my data")
    r = client.get("/api/tickets", headers=auth_header)
    assert r.status_code == 200
    ticket = next(t for t in r.json() if t["id"] == tid)
    assert ticket["category"] == "ACCOUNT"
    assert ticket["priority"] in {"low", "medium", "high"}
    assert ticket["status"] == "open"


def test_filter_by_category(client, auth_header, make_ticket):
    make_ticket("i want to subscribe to your newsletter")
    make_ticket("how do i delete my account and my data")
    r = client.get("/api/tickets", params={"category": "SUBSCRIPTION"}, headers=auth_header)
    assert r.status_code == 200
    assert r.json() and all(t["category"] == "SUBSCRIPTION" for t in r.json())


def test_unknown_filter_values_are_422(client, auth_header):
    assert client.get("/api/tickets", params={"category": "NOPE"}, headers=auth_header).status_code == 422
    assert client.get("/api/tickets", params={"status": "NOPE"}, headers=auth_header).status_code == 422


def test_change_status(client, auth_header, make_ticket):
    tid = make_ticket()
    r = client.patch(f"/api/tickets/{tid}", json={"status": "in_progress"}, headers=auth_header)
    assert r.status_code == 200 and r.json()["status"] == "in_progress"
    assert client.get(f"/api/tickets/{tid}", headers=auth_header).json()["status"] == "in_progress"


def test_invalid_status_and_missing_ticket(client, auth_header, make_ticket):
    tid = make_ticket()
    assert client.patch(f"/api/tickets/{tid}", json={"status": "banana"}, headers=auth_header).status_code == 422
    assert client.patch("/api/tickets/999999", json={"status": "closed"}, headers=auth_header).status_code == 404
    assert client.get("/api/tickets/999999", headers=auth_header).status_code == 404


def test_draft_is_attached_with_faq_entries(client, auth_header, make_ticket):
    tid = make_ticket("do you accept paypal")
    t = client.get(f"/api/tickets/{tid}", headers=auth_header).json()
    assert t["draft_status"] == "ready"  # background task has run
    assert t["draft_grounded"] is True
    assert t["draft_faq_ids"] and t["draft_faq_entries"][0]["id"] == t["draft_faq_ids"][0]


def test_unanswerable_ticket_gets_honest_no_match_draft(client, auth_header, make_ticket):
    tid = make_ticket("what is the warranty on the blender i bought")
    t = client.get(f"/api/tickets/{tid}", headers=auth_header).json()
    assert t["draft_grounded"] is False and t["draft_faq_ids"] == []
    assert "could not find an answer" in t["draft_text"]


def test_hostile_markup_is_stored_verbatim_not_executed(client, auth_header, make_ticket):
    payload = "<script>alert(1)</script> please help me with my order"
    tid = make_ticket(payload)
    assert client.get(f"/api/tickets/{tid}", headers=auth_header).json()["message"] == payload  # UI uses textContent


def test_pages_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/team").status_code == 200
