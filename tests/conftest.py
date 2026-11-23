import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before `app` is imported: a throwaway database, a fixed login, no real LLM.
_tmp = tempfile.mkdtemp(prefix="support-desk-tests-")
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["ADMIN_USER"] = "agent"
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["LLM_PROVIDER"] = "none"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:  # runs lifespan: creates tables, loads the model
        yield c


@pytest.fixture(scope="session")
def auth_header(client):
    r = client.post("/api/login", json={"username": "agent", "password": "test-password"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


@pytest.fixture()
def make_ticket(client):
    def _make(message="where is my parcel? it has not arrived", name="Ann", email="ann@example.com"):
        r = client.post("/api/tickets", json={"name": name, "email": email, "message": message})
        assert r.status_code == 201, r.text
        return r.json()["id"]

    return _make
