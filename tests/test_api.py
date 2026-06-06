"""Offline tests for the FastAPI backend (no OPENAI_API_KEY, no Redis server)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verification_agents.api import app

BUG = (
    "def get_item(items, index):\n"
    "    for index in range(len(items) + 1):\n"
    "        value = items[index]\n"
    "    return value\n"
)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # force offline heuristic path
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["redis"] in ("redis", "memory")


def test_analyze_sync_returns_full_report(client):
    r = client.post("/api/analyze", json={"code": BUG})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "no"
    assert body["job_id"]
    assert any(b["concern"] == "array_bounds" for b in body["bugs"])
    # streamed events were recorded for the UI
    stages = [e["stage"] for e in body["events"]]
    assert "start" in stages and "done" in stages


def test_job_refetch(client):
    job_id = client.post("/api/analyze", json={"code": BUG}).json()["job_id"]
    again = client.get(f"/api/jobs/{job_id}").json()
    assert again["decision"] == "no"
    events = client.get(f"/api/jobs/{job_id}/events").json()["events"]
    assert events and events[-1]["stage"] == "done"


def test_analyze_requires_code_or_diff(client):
    r = client.post("/api/analyze", json={"repo_url": "https://github.com/x/y"})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
