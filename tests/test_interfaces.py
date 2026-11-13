"""CLI, web app, exports and the MCP server."""
import json
import os
import subprocess
import sys
import time

import pytest

from conftest import SEGYIO, needs_segyio_data

CLI = [sys.executable, "-m", "segwise"]


def run(*args):
    return subprocess.run(CLI + list(args), capture_output=True, text=True)


def test_cli_scan_exit_codes(torture_dir, tmp_path):
    p = run("scan", torture_dir["standard"][1], "-o", str(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "READY" in p.stdout and (tmp_path / "standard.report.html").exists()
    ls = json.loads((tmp_path / "standard.loadsheet.json").read_text())
    assert ls["schema"] == "segwise.loadsheet/1" and ls["geometry"]["inline"]["byte"] == 189
    assert run("scan", torture_dir["no_text"][1], "--no-files").returncode == 1  # CRS needs a decision
    bad = tmp_path / "bad.sgy"
    bad.write_bytes(b"x" * 100)
    assert run("scan", str(bad), "--no-files").returncode == 2


def test_cli_json_and_overrides(torture_dir):
    p = run("scan", torture_dir["legacy_9_21_73_77"][1], "--json", "--crs", "EPSG:23031")
    d = json.loads(p.stdout)
    assert d["geometry"]["inline"]["byte"] == 9 and d["crs"]["chosen"]["epsg"] == 23031


def test_cli_fix_and_loadsheet_reuse(torture_dir, tmp_path):
    src = torture_dir["legacy_9_21_73_77"][1]
    out = tmp_path / "fixed.sgy"
    p = run("fix", src, "-o", str(out), "--crs", "23031")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PASS" in p.stdout and out.exists()
    sheet = tmp_path / "fixed.loadsheet.json"
    assert sheet.exists()
    # apply the confirmed sheet to a sibling file of the same delivery
    p2 = run("scan", src, "--loadsheet", str(sheet), "--json")
    assert json.loads(p2.stdout)["crs"]["status"] == "confirmed"


def test_cli_headers_csv(torture_dir, tmp_path):
    p = run("headers", torture_dir["standard"][1], "--bytes", "189,193,181,185,71", "--csv", str(tmp_path / "h.csv"))
    assert p.returncode == 0
    lines = (tmp_path / "h.csv").read_text().splitlines()
    assert lines[0].startswith("trace,189") and len(lines) == 21


def test_inventory(torture_dir, tmp_path):
    from segwise.inventory import run_inventory

    d = os.path.dirname(torture_dir["standard"][1])
    res = run_inventory(d, str(tmp_path / "inv"), jobs=2)
    assert res["n_files"] >= 26
    out = tmp_path / "inv"
    for f in ("inventory.csv", "inventory.geojson", "index.html", "groups.json"):
        assert (out / f).exists()
    gj = json.loads((out / "inventory.geojson").read_text())
    assert any(f["geometry"]["type"] == "Polygon" for f in gj["features"])


def test_report_is_self_contained(torture_dir):
    from segwise.analyze import analyze
    from segwise.report_html import render_report

    html = render_report(analyze(torture_dir["wrong_scalar"][1]))
    assert "<script src" not in html and "<link" not in html and "http://" not in html.split("<body")[0]
    assert "data:image/png;base64" in html and "How to load it" in html and "Problems found" in html


def test_web_app_flow(torture_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SEGWISE_HOME", str(tmp_path / "home"))
    import importlib

    import segwise.web.app as webapp

    importlib.reload(webapp)
    from fastapi.testclient import TestClient

    c = TestClient(webapp.create_app(os.path.dirname(torture_dir["standard"][1])))
    H = {"X-Segwise": "1"}
    assert c.get("/").status_code == 200
    assert c.post("/api/scan", json={"path": torture_dir["standard"][1]}).status_code == 403  # CSRF guard

    def wait(job):
        for _ in range(300):
            s = c.get(f"/api/jobs/{job}").json()
            if s["done"]:
                assert s["error"] is None, s["error"]
                return s["result"]
            time.sleep(0.1)
        raise AssertionError("job timed out")

    res = wait(c.post("/api/scan", json={"path": torture_dir["wrong_scalar"][1]}, headers=H).json()["job"])
    assert c.get(res["redirect"]).status_code == 200
    rid = res["id"]
    res = wait(c.post(f"/api/file/{rid}/fix", json={"output": str(tmp_path / "w.sgy")}, headers=H).json()["job"])
    assert res["passed"] and os.path.exists(tmp_path / "w.sgy")
    assert c.post(f"/api/file/{rid}/fix", json={"output": torture_dir["wrong_scalar"][1]}, headers=H).status_code == 400
    assert c.get(f"/api/file/{rid}/loadsheet").json()["schema"] == "segwise.loadsheet/1"
    for page in ("/inventory", "/loadsheets", "/nav", "/settings"):
        assert c.get(page).status_code == 200


def test_mcp_tools(torture_dir):
    pytest.importorskip("mcp")
    import asyncio
    import json as _json

    from segwise.mcp_server import server

    async def go():
        names = {t.name for t in await server.list_tools()}
        assert {"scan_segy", "fix_segy", "inventory_segy", "header_values", "merge_navigation"} <= names
        res = await server.call_tool("scan_segy", {"path": torture_dir["legacy_9_21_73_77"][1]})
        text = res.content[0].text if hasattr(res, "content") else res[0].text
        d = _json.loads(text)
        assert d["geometry"]["inline"]["byte"] == 9 and d["status"] in ("fixable", "needs_review")

    asyncio.run(go())


def test_web_no_silent_overwrite_and_host_check(torture_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SEGWISE_HOME", str(tmp_path / "home"))
    import importlib

    import segwise.web.app as webapp

    importlib.reload(webapp)
    from fastapi.testclient import TestClient

    c = TestClient(webapp.create_app(str(tmp_path)))
    H = {"X-Segwise": "1"}
    assert c.get("/", headers={"host": "evil.example.com"}).status_code == 403
    jid = c.post("/api/scan", json={"path": torture_dir["standard"][1]}, headers=H).json()["job"]
    for _ in range(300):
        s = c.get(f"/api/jobs/{jid}").json()
        if s["done"]:
            break
        time.sleep(0.1)
    rid = s["result"]["id"]
    victim = tmp_path / "existing.sgy"
    victim.write_bytes(b"precious")
    r = c.post(f"/api/file/{rid}/fix", json={"output": str(victim)}, headers=H)
    assert r.status_code == 409 and victim.read_bytes() == b"precious"
    r = c.post(f"/api/file/{rid}/fix", json={"output": "rel_out.sgy"}, headers=H)
    assert r.status_code == 200
    jid = r.json()["job"]
    for _ in range(300):
        if c.get(f"/api/jobs/{jid}").json()["done"]:
            break
        time.sleep(0.1)
    assert (tmp_path / "rel_out.sgy").exists()  # relative paths resolve against the app's folder


def test_one_place_for_files_and_folders_with_native_picker(torture_dir, tmp_path, monkeypatch):
    """Files and folders are checked from the same home page; the Choose buttons use the OS dialog."""
    monkeypatch.setenv("SEGWISE_HOME", str(tmp_path / "home"))
    import importlib

    import segwise.picker as picker
    from segwise.web import app as webapp

    importlib.reload(webapp)
    from fastapi.testclient import TestClient

    folder = os.path.dirname(torture_dir["standard"][1])
    calls = []

    def fake_pick(kind, start=None):
        calls.append((kind, start))
        return {"folder": folder, "file": torture_dir["standard"][1]}.get(kind)

    monkeypatch.setattr(picker, "pick", fake_pick)
    c = TestClient(webapp.create_app(str(tmp_path)))
    H = {"X-Segwise": "1"}
    home = c.get("/").text
    assert 'data-pick="file"' in home and 'data-pick="folder"' in home and "Deliveries" not in home
    assert c.post("/api/pick", json={"kind": "folder"}).status_code == 403  # CSRF guard
    assert c.post("/api/pick", json={"kind": "bogus"}, headers=H).status_code == 400
    r = c.post("/api/pick", json={"kind": "folder"}, headers=H).json()
    assert r["path"] == folder and r["exists"] is True and calls[-1][0] == "folder"

    def wait(job):
        for _ in range(600):
            s = c.get(f"/api/jobs/{job}").json()
            if s["done"]:
                assert s["error"] is None, s["error"]
                return s["result"]
            time.sleep(0.1)
        raise AssertionError("job timed out")

    res = wait(c.post("/api/scan", json={"path": folder}, headers=H).json()["job"])
    assert res["redirect"].startswith("/inventory/")
    home = c.get("/").text
    assert "Recent checks" in home and res["redirect"] in home  # the folder is listed with the files
    assert c.get("/inventory", follow_redirects=False).status_code == 307
    # after a restart the folder link re-checks the folder instead of failing
    c2 = TestClient(webapp.create_app(str(tmp_path)))
    assert "Re-checking" in c2.get(res["redirect"]).text

    def unavailable(kind, start=None):
        raise picker.PickerUnavailable("no display")

    monkeypatch.setattr(picker, "pick", unavailable)
    r = c.post("/api/pick", json={"kind": "file"}, headers=H)
    assert r.status_code == 501 and "Paste the path" in r.json()["detail"]
