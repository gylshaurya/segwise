"""Batch mode: scan a whole data delivery.

Outputs (in the chosen folder):
  inventory.csv          one row per file - drop into Excel or a data catalogue
  inventory.geojson      survey footprints (WGS 84) - drag into QGIS/ArcGIS
  index.html             browsable summary with status, layout groups and links
  reports/<file>.html    per-file reports; reports/<file>.loadsheet.json
  groups.json            files sharing one header layout (confirm once, apply to all)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import traceback
from collections import Counter, defaultdict

SEGY_EXT = {".sgy", ".segy", ".seg", ".sgy2", ".su"}


def _looks_like_segy(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size < 3600:
            return False
        from .reader import _endian_score, parse_binary_header

        with open(path, "rb") as f:
            f.seek(3200)
            b = f.read(400)
        return max(_endian_score(parse_binary_header(b, e)) for e in (">", "<")) >= 5.5
    except Exception:
        return False


def find_segy(root: str, recursive: bool = True) -> list[str]:
    out = []
    walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
    for d, dirs, files in walker:
        dirs[:] = [x for x in dirs if not x.startswith(".") and not x.startswith("segwise")]
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            p = os.path.join(d, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in SEGY_EXT or (ext in ("", ".dat", ".bin", ".raw") and _looks_like_segy(p)):
                out.append(p)
    return sorted(out)


def _one(args):
    path, overrides, out_dir = args
    from .analyze import analyze
    from .exports import geojson, loadsheet, write_json
    from .report_html import render_report

    try:
        rep = analyze(path, overrides=overrides)
        stem = _safe_stem(path)
        rd = os.path.join(out_dir, "reports")
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, f"{stem}.html"), "w") as f:
            f.write(render_report(rep, back_link="../index.html"))
        write_json(os.path.join(rd, f"{stem}.loadsheet.json"), loadsheet(rep))
        return {"ok": True, "report": rep, "stem": stem, "geojson": geojson(rep)}
    except Exception as e:  # one broken file must never stop a batch
        return {"ok": False, "path": path, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    s = f"{parent}__{stem}" if parent else stem
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def run_inventory(directory: str, output: str, jobs: int = 4, recursive: bool = True, overrides=None,
                  progress: bool = False) -> dict:
    from .exports import INVENTORY_COLUMNS, inventory_csv, inventory_row, write_json
    from .report_html import render_inventory

    files = find_segy(directory, recursive)
    os.makedirs(output, exist_ok=True)
    reports, failures, feats = [], [], []
    args = [(p, overrides, output) for p in files]
    done = 0
    pool = cf.ProcessPoolExecutor(max_workers=jobs) if jobs > 1 and len(files) > 1 else None
    try:
        it = pool.map(_one, args) if pool else map(_one, args)
        for res in it:
            done += 1
            if progress and sys.stderr.isatty():
                sys.stderr.write(f"\r  scanned {done}/{len(files)}")
                sys.stderr.flush()
            if res["ok"]:
                reports.append((res["report"], res["stem"]))
                feats += res["geojson"]["features"]
            else:
                failures.append(res)
    finally:
        if pool:
            pool.shutdown()
    if progress and sys.stderr.isatty():
        sys.stderr.write("\n")
    reps = [r for r, _ in reports]
    with open(os.path.join(output, "inventory.csv"), "w", newline="") as f:
        f.write(inventory_csv(reps))
    write_json(os.path.join(output, "inventory.geojson"), {"type": "FeatureCollection", "features": feats})
    groups = defaultdict(list)
    for r in reps:
        groups[r.layout_key].append(r.file_name)
    write_json(os.path.join(output, "groups.json"), groups)
    by_status = Counter(r.status for r in reps)
    if failures:
        by_status["unusable"] += len(failures)
    with open(os.path.join(output, "index.html"), "w") as f:
        f.write(render_inventory(reports, failures, directory, dict(groups)))
    if failures:
        write_json(os.path.join(output, "failures.json"), [{k: v for k, v in x.items() if k != "ok"} for x in failures])
    return {"n_files": len(files), "output": os.path.abspath(output), "by_status": dict(by_status),
            "n_groups": len(groups), "failures": failures}
