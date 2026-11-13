"""Local web app: `segwise ui`.

Runs on 127.0.0.1 only. Files are read in place from local/network paths (no
uploads - seismic files are often tens of GB), nothing leaves the machine.
State-changing requests must carry the X-Segwise header, which browsers
will not let other websites add (simple CSRF protection for a local server).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import json
import os
import threading
import time
import traceback
import uuid
import webbrowser
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from .. import __version__
from ..report_html import context, css, env, figures, js

HOME = os.path.expanduser(os.environ.get("SEGWISE_HOME", "~/.segwise"))


class State:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.reports: dict[str, Any] = {}  # id -> Report
        self.figs: dict[str, dict] = {}
        self.fixlogs: dict[str, Any] = {}
        self.inventories: dict[str, dict] = {}
        self.file_inv: dict[str, str] = {}  # file id -> delivery it was checked in
        self.jobs: dict[str, dict] = {}
        self.pool = cf.ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        os.makedirs(HOME, exist_ok=True)
        self.history_path = os.path.join(HOME, "history.json")
        self.settings_path = os.path.join(HOME, "settings.json")
        self.history: list[dict] = self._load(self.history_path, [])
        self.settings: dict = self._load(self.settings_path, {"out_dir": os.path.join(self.root, "segwise_out")})

    @staticmethod
    def _load(path, default):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default

    def save(self):
        with self.lock:
            for path, obj in ((self.history_path, self.history[:200]), (self.settings_path, self.settings)):
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(obj, f, indent=1)
                os.replace(tmp, path)

    def remember(self, rid: str, rep) -> None:
        self.history = [h for h in self.history if h.get("path") != rep.path]
        self.history.insert(0, {"kind": "file", "id": rid, "path": rep.path, "file_name": rep.file_name,
                                "status": rep.status, "summary": rep.summary, "reason": rep.status_reason,
                                "at": time.time()})
        self.save()

    def remember_folder(self, inv_id: str, path: str) -> None:
        inv = self.inventories[inv_id]
        counts: dict[str, int] = {}
        for it in inv["items"]:
            s = self.reports[it["id"]].status if "id" in it else "unusable"
            counts[s] = counts.get(s, 0) + 1
        worst = next((s for s in ("unusable", "needs_review", "fixable", "ready") if counts.get(s)), "ready")
        self.history = [h for h in self.history if h.get("path") != path]
        self.history.insert(0, {"kind": "folder", "id": inv_id, "path": path,
                                "file_name": os.path.basename(path.rstrip("/\\")) or path, "status": worst,
                                "counts": counts, "n": len(inv["items"]), "at": time.time()})
        self.save()


def ago(t: float) -> str:
    s = max(0, time.time() - t)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def create_app(root: str | None = None) -> FastAPI:
    st = State(root or os.getcwd())
    app = FastAPI(title="segwise", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.st = st
    E = env()
    allowed_hosts = {"127.0.0.1", "localhost", "::1", "[::1]", "testserver"}

    @app.middleware("http")
    async def only_local(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not (request.headers.get("host") or "").startswith("[") \
            else (request.headers.get("host") or "").split("]")[0] + "]"
        if host not in allowed_hosts:
            return PlainTextResponse("Segwise only answers requests addressed to this computer.", status_code=403)
        return await call_next(request)

    def page(tpl: str, request: Request, **kw) -> HTMLResponse:
        base = {"css": css(), "js": js(), "version": __version__, "path": request.url.path, "history": st.history[:30],
                "ago": ago, "settings": st.settings,
                "theme": request.query_params.get("theme") or st.settings.get("theme", "auto")}
        return HTMLResponse(E.get_template(tpl).render(**base, **kw))

    def guard(request: Request):
        if request.headers.get("x-segwise") != "1":
            raise HTTPException(403, "Missing X-Segwise header (requests must come from the Segwise app).")

    def start_job(kind: str, fn, *args) -> str:
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "stage": "Queued", "progress": 0.0, "done": False, "error": None, "result": None,
               "started": time.time()}
        st.jobs[jid] = job

        def progress(stage: str, frac: float):
            job["stage"], job["progress"] = stage, round(float(frac), 3)

        def run():
            try:
                job["result"] = fn(progress, *args)
            except Exception as e:
                job["error"] = f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()
            finally:
                job["done"] = True
                job["progress"] = 1.0

        st.pool.submit(run)
        return jid

    # ------------------------------------------------------------------ work functions
    def do_scan(progress, path: str, overrides: dict, rid: str | None = None):
        from ..analyze import analyze

        rep = analyze(path, overrides=overrides, progress=progress)
        rid = rid or uuid.uuid4().hex[:10]
        progress("Drawing figures", 0.95)
        st.figs[rid] = figures(rep)
        st.reports[rid] = rep
        st.remember(rid, rep)
        return {"redirect": f"/file/{rid}", "id": rid}

    def do_inventory(progress, path: str, overrides: dict, inv_id: str | None = None):
        from ..analyze import analyze
        from ..inventory import find_segy

        files_ = find_segy(path)
        if not files_:
            raise ValueError(f"No SEG-Y files found in {path}")
        inv_id = inv_id or uuid.uuid4().hex[:10]
        items = []
        st.inventories[inv_id] = {"path": path, "items": items, "total": len(files_), "started": time.time()}
        for k, fp in enumerate(files_):
            progress(f"Scanning {os.path.basename(fp)} ({k + 1}/{len(files_)})", k / len(files_))
            try:
                rep = analyze(fp, overrides=overrides)
                rid = uuid.uuid4().hex[:10]
                st.reports[rid] = rep
                st.file_inv[rid] = inv_id
                items.append({"id": rid})
            except Exception as e:
                items.append({"error": f"{type(e).__name__}: {e}", "path": fp})
        st.remember_folder(inv_id, path)
        return {"redirect": f"/inventory/{inv_id}", "id": inv_id}

    def do_fix(progress, rid: str, out: str, ieee: bool, nav: dict | None, overwrite: bool = False):
        from ..analyze import analyze
        from ..writer import write_fixed

        rep = st.reports[rid]
        navm = None
        if nav and nav.get("path"):
            from ..nav import match, read_nav

            navm = match(rep, read_nav(nav["path"]), line=nav.get("line") or None,
                         to_epsg=int(nav["to_epsg"]) if nav.get("to_epsg") else None,
                         nav_epsg=int(nav["nav_epsg"]) if nav.get("nav_epsg") else None)
        log = write_fixed(rep, out, ieee=ieee, nav=navm.as_writer_nav() if navm else None, progress=progress,
                          overwrite=overwrite)
        new = analyze(out, overrides={"epsg": navm.to_epsg} if navm and navm.to_epsg else (
            {"epsg": rep.crs["chosen"]["epsg"]} if rep.crs.get("chosen") and rep.crs.get("status") == "confirmed" else None))
        if navm:
            new.notes = navm.notes + new.notes
        nid = uuid.uuid4().hex[:10]
        st.reports[nid] = new
        st.fixlogs[nid] = log
        st.figs[nid] = figures(new)
        st.remember(nid, new)
        # also drop the load sheet + standalone report next to the fixed file
        from ..cli import _write_outputs

        _write_outputs(new, os.path.dirname(out), fixlog=log, stem=os.path.splitext(os.path.basename(out))[0])
        return {"redirect": f"/file/{nid}", "id": nid, "passed": log.verification.get("passed")}

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return page("web_home.html", request, root=st.root)

    @app.get("/file/{rid}", response_class=HTMLResponse)
    def file_page(request: Request, rid: str):
        rep = st.reports.get(rid)
        if rep is None:
            h = next((h for h in st.history if h["id"] == rid), None)
            if h and os.path.exists(h["path"]):
                jid = start_job("scan", do_scan, h["path"], {}, rid)
                return page("web_job.html", request, job_id=jid, title=f"Re-scanning {h['file_name']}")
            raise HTTPException(404, "Unknown file. Scan it again from the home page.")
        figs = st.figs.get(rid)
        if figs is None:
            figs = st.figs[rid] = figures(rep)
        inv_id = st.file_inv.get(rid)
        ctx = context(rep, st.fixlogs.get(rid), back_link=f"/inventory/{inv_id}" if inv_id else "/", figs=figs,
                      embed_loadsheet=False)
        ctx["back_label"] = "Back to the delivery" if inv_id else "Check another file"
        ctx["loadsheet_href"] = f"/api/file/{rid}/loadsheet"
        ctx["report_href"] = f"/api/file/{rid}/report.html"
        from ..exports import reusable_overrides

        stem = os.path.splitext(rep.file_name)[0]
        default_out = os.path.join(os.path.dirname(rep.path), f"{stem}_std.sgy")
        if not os.access(os.path.dirname(rep.path) or ".", os.W_OK):
            default_out = os.path.join(st.settings.get("out_dir"), f"{stem}_std.sgy")
        cand_map = rep.header_map
        return page("web_file.html", request, rid=rid, default_out=default_out, decisions=reusable_overrides(rep),
                    header_options=[h for h in cand_map if not h["constant"]], **ctx)

    @app.get("/inventory")
    def inventory_list():
        return RedirectResponse("/", status_code=307)  # files and folders are checked from the same place

    @app.get("/inventory/{inv_id}", response_class=HTMLResponse)
    def inventory_page(request: Request, inv_id: str):
        from ..exports import inventory_row

        inv = st.inventories.get(inv_id)
        if inv is None:
            h = next((h for h in st.history if h["id"] == inv_id and h.get("kind") == "folder"), None)
            if h and os.path.isdir(h["path"]):
                jid = start_job("inventory", do_inventory, h["path"], {}, inv_id)
                return page("web_job.html", request, job_id=jid, title=f"Re-checking {h['file_name']}")
            raise HTTPException(404, "Unknown folder. Check it again from the home page.")
        rows, groups, counts = [], {}, {}
        for it in inv["items"]:
            if "id" in it:
                rep = st.reports[it["id"]]
                row = inventory_row(rep)
                rows.append((row, it["id"]))
                groups.setdefault(rep.layout_key, []).append(rep.file_name)
                counts[rep.status] = counts.get(rep.status, 0) + 1
            else:
                counts["unusable"] = counts.get("unusable", 0) + 1
        order = {"unusable": 0, "needs_review": 1, "fixable": 2, "ready": 3}
        rows.sort(key=lambda t: (order.get(t[0]["status"], 9), t[0]["file_name"]))
        failures = [it for it in inv["items"] if "error" in it]
        return page("web_inventory.html", request, inv=inv, inv_id=inv_id, rows=rows, groups=groups, counts=counts,
                    failures=failures, root=st.root, all_inv=[])

    @app.get("/job/{jid}", response_class=HTMLResponse)
    def job_page(request: Request, jid: str):
        if jid not in st.jobs:
            raise HTTPException(404, "Unknown job")
        return page("web_job.html", request, job_id=jid, title="Working...")

    @app.get("/nav", response_class=HTMLResponse)
    def nav_page(request: Request):
        return page("web_nav.html", request, files=[(rid, r) for rid, r in st.reports.items()][-50:])

    @app.get("/loadsheets", response_class=HTMLResponse)
    def loadsheets_page(request: Request):
        from ..exports import reusable_overrides

        sheets = [(rid, r, reusable_overrides(r)) for rid, r in st.reports.items() if r.status != "unusable"]
        return page("web_loadsheets.html", request, sheets=sheets[::-1])

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return page("web_settings.html", request)

    # ------------------------------------------------------------------ API
    @app.post("/api/scan")
    async def api_scan(request: Request):
        guard(request)
        body = await request.json()
        path = os.path.expanduser(str(body.get("path", "")).strip().strip('"').strip("'"))
        if not path:
            raise HTTPException(400, "Enter the path of a SEG-Y file or a folder.")
        if not os.path.isabs(path):
            path = os.path.join(st.root, path)
        if not os.path.exists(path):
            raise HTTPException(404, f"Nothing found at {path}")
        ov = body.get("overrides") or {}
        if os.path.isdir(path):
            jid = start_job("inventory", do_inventory, path, ov)
        else:
            jid = start_job("scan", do_scan, path, ov)
        return {"job": jid}

    @app.post("/api/file/{rid}/reanalyze")
    async def api_reanalyze(request: Request, rid: str):
        guard(request)
        rep = st.reports.get(rid)
        if rep is None:
            raise HTTPException(404, "Unknown file")
        ov = (await request.json()).get("overrides") or {}
        jid = start_job("scan", do_scan, rep.path, ov, rid)
        st.fixlogs.pop(rid, None)
        return {"job": jid}

    @app.post("/api/file/{rid}/fix")
    async def api_fix(request: Request, rid: str):
        guard(request)
        if rid not in st.reports:
            raise HTTPException(404, "Unknown file")
        body = await request.json()
        out = os.path.expanduser(str(body.get("output", "")).strip())
        if not out:
            raise HTTPException(400, "Choose where to write the fixed file.")
        if not os.path.isabs(out):
            out = os.path.join(st.root, out)
        if os.path.exists(out) and os.path.samefile(out, st.reports[rid].path):
            raise HTTPException(400, "The output must be a new file; the original is never overwritten.")
        if os.path.exists(out) and not body.get("overwrite"):
            raise HTTPException(409, f"{out} already exists. Tick 'Replace existing file' to overwrite it, or choose another name.")
        if any(i.code == "STRUCTURE_BREAK" for i in st.reports[rid].issues):
            raise HTTPException(400, "This file's structure breaks part-way through; split it before writing a fixed copy.")
        jid = start_job("fix", do_fix, rid, out, bool(body.get("ieee")), body.get("nav"), bool(body.get("overwrite")))
        return {"job": jid}

    @app.post("/api/inventory/{inv_id}/apply")
    async def api_inv_apply(request: Request, inv_id: str):
        guard(request)
        inv = st.inventories.get(inv_id)
        if inv is None:
            raise HTTPException(404, "Unknown inventory")
        ov = (await request.json()).get("overrides") or {}

        def redo(progress):
            from ..analyze import analyze

            items = [it for it in inv["items"] if "id" in it]
            for k, it in enumerate(items):
                rep = st.reports[it["id"]]
                progress(f"Re-checking {rep.file_name} ({k + 1}/{len(items)})", k / max(len(items), 1))
                merged = dict(rep.overrides)
                merged.update(ov)
                new = analyze(rep.path, overrides=merged)
                st.reports[it["id"]] = new
                st.figs.pop(it["id"], None)
            inv["overrides"] = ov
            st.remember_folder(inv_id, inv["path"])
            return {"redirect": f"/inventory/{inv_id}"}

        return {"job": start_job("inventory", redo)}

    @app.post("/api/settings")
    async def api_settings(request: Request):
        guard(request)
        body = await request.json()
        if body.get("theme") in ("auto", "light", "dark"):
            st.settings["theme"] = body["theme"]
        if body.get("out_dir"):
            st.settings["out_dir"] = os.path.expanduser(body["out_dir"])
        st.save()
        return st.settings

    @app.post("/api/pick")
    async def api_pick(request: Request):
        """Show the operating system's own file/folder dialog and return the chosen path."""
        import asyncio

        from ..picker import KINDS, PickerUnavailable, pick

        guard(request)
        body = await request.json()
        kind = body.get("kind", "file")
        if kind not in KINDS:
            raise HTTPException(400, "Unknown picker kind")
        start = body.get("start") or st.root
        try:
            chosen = await asyncio.to_thread(pick, kind, start)
        except PickerUnavailable as e:
            raise HTTPException(501, f"This computer has no file dialog available ({e}). Paste the path instead.")
        return {"path": chosen, "exists": bool(chosen and os.path.exists(chosen))}

    @app.post("/api/history/clear")
    async def api_clear(request: Request):
        guard(request)
        st.history = []
        st.save()
        return {"ok": True}

    @app.get("/api/jobs/{jid}")
    def api_job(jid: str):
        j = st.jobs.get(jid)
        if j is None:
            raise HTTPException(404, "Unknown job")
        return {k: v for k, v in j.items() if k != "trace"}

    @app.get("/api/file/{rid}/loadsheet")
    def api_loadsheet(rid: str):
        from ..exports import loadsheet

        rep = st.reports.get(rid)
        if rep is None:
            raise HTTPException(404, "Unknown file")
        data = json.dumps(loadsheet(rep, st.fixlogs.get(rid)), indent=2)
        name = os.path.splitext(rep.file_name)[0] + ".loadsheet.json"
        return Response(data, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/file/{rid}/report.html")
    def api_report(rid: str):
        from ..report_html import render_report

        rep = st.reports.get(rid)
        if rep is None:
            raise HTTPException(404, "Unknown file")
        name = os.path.splitext(rep.file_name)[0] + ".report.html"
        return Response(render_report(rep, st.fixlogs.get(rid)), media_type="text/html",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/file/{rid}/headers")
    def api_headers(rid: str, bytes: str = "1,5,9,21,71,73,77,181,185,189,193", start: int = 0, n: int = 50):
        import numpy as np

        from ..fields import TRACE_FIELD_BY_BYTE
        from ..reader import TraceReader, probe

        rep = st.reports.get(rid)
        if rep is None:
            raise HTTPException(404, "Unknown file")
        lay = probe(rep.path)
        r = TraceReader(lay)
        idx = np.arange(max(0, start), min(lay.n_traces, start + min(n, 500)))
        cols = {}
        for tok in bytes.split(","):
            b, _, t = tok.partition(":")
            b = int(b)
            f = TRACE_FIELD_BY_BYTE.get(b)
            t = t or ("int16" if f and f.size == 2 else "int32")
            cols[f"{b}:{t}"] = r.field(idx, b, t).tolist()
        return {"trace": idx.tolist(), "columns": cols}

    @app.get("/api/browse")
    def api_browse(path: str = ""):
        p = os.path.expanduser(path) if path else st.root
        if not os.path.isdir(p):
            p = os.path.dirname(p) or st.root
        try:
            entries = sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            raise HTTPException(403, f"No permission to read {p}")
        out = []
        for e in entries[:500]:
            if e.name.startswith("."):
                continue
            ext = os.path.splitext(e.name)[1].lower()
            if e.is_dir() or ext in (".sgy", ".segy", ".seg", ".su", ".sgy2"):
                out.append({"name": e.name, "path": e.path, "dir": e.is_dir(),
                            "size": e.stat().st_size if not e.is_dir() else None})
        return {"path": p, "parent": os.path.dirname(p.rstrip("/")) or "/", "entries": out}

    @app.get("/healthz")
    def health():
        return PlainTextResponse("segwise")

    return app


def _running_here(host: str, port: int) -> str:
    """'segwise' if segwise already answers on this port, 'busy' if something else does, '' if free."""
    import socket
    import urllib.request

    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) != 0:
            return ""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=1) as r:
            return "segwise" if r.read().decode().strip() == "segwise" else "busy"
    except Exception:
        return "busy"


def serve(host: str = "127.0.0.1", port: int = 8765, root: str | None = None, open_browser: bool = True) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Warning: serving on a non-local address exposes your files to the network.")
    for p in range(port, port + 20):
        state = _running_here(host, p)
        if state == "segwise":  # already open (e.g. double-clicked twice): just show it
            url = f"http://{host}:{p}/"
            print(f"Segwise is already running at {url}")
            if open_browser:
                webbrowser.open(url)
            return
        if state == "":
            port = p
            break
    url = f"http://{host}:{port}/"
    print(f"\n  Segwise {__version__} is running at {url}\n"
          "  Your browser should open by itself. If not, copy the address above into it.\n"
          "  Keep this window open while you use Segwise; close it (or press Ctrl+C) to quit.\n"
          "  Your files never leave this computer.\n", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")
