"""Command-line interface.

    segwise scan FILE...            analyse, print a verdict, write report + load sheet
    segwise fix FILE -o OUT.sgy     write a standardised, verified copy
    segwise inventory DIR           batch-scan a delivery: CSV, GeoJSON, HTML index
    segwise headers FILE            dump trace-header values
    segwise nav FILE --nav NAV      merge navigation into a file without coordinates
    segwise ui                      local web app (data never leaves this machine)

Exit codes (for scripts and ingestion pipelines):
    0  ready or fixable automatically
    1  needs a human decision (e.g. CRS ambiguous, no coordinates)
    2  unusable / error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__
from .model import FieldRef

STATUS_EXIT = {"ready": 0, "fixable": 0, "needs_review": 1, "unusable": 2}


# ------------------------------------------------------------------ terminal styling
class _C:
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, s: str, code: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s):
        return self(s, "1")

    def dim(self, s):
        return self(s, "2")

    def red(self, s):
        return self(s, "31")

    def green(self, s):
        return self(s, "32")

    def yellow(self, s):
        return self(s, "33")

    def blue(self, s):
        return self(s, "34")


C = _C(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)

STATUS_LABEL = {"ready": ("READY", "green"), "fixable": ("FIXABLE", "blue"), "needs_review": ("NEEDS A DECISION", "yellow"),
                "unusable": ("CAN'T LOAD", "red")}


def _status(s: str) -> str:
    label, col = STATUS_LABEL.get(s, (s.upper(), "1"))
    return C.bold(getattr(C, col)(f"[{label}]"))


# ------------------------------------------------------------------ argument helpers
def _ref(s: str | None, default_type: str = "int32") -> FieldRef | None:
    if not s:
        return None
    if ":" in s:
        b, t = s.split(":", 1)
        t = {"i4": "int32", "int": "int32", "i2": "int16", "short": "int16", "f4": "float32", "float": "float32"}.get(t, t)
        return FieldRef(int(b), t)
    return FieldRef(int(s), default_type)


def _epsg(s: str | None) -> int | None:
    if not s:
        return None
    return int(str(s).upper().replace("EPSG:", "").strip())


def add_override_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("overrides (each one wins over automatic inference)")
    g.add_argument("--inline", metavar="BYTE[:TYPE]", help="inline byte, e.g. 9 or 9:int16")
    g.add_argument("--crossline", "--xline", dest="crossline", metavar="BYTE[:TYPE]", help="crossline byte")
    g.add_argument("--cdp", metavar="BYTE[:TYPE]", help="2D CDP/trace key byte")
    g.add_argument("--x", metavar="BYTE[:TYPE]", help="X coordinate byte (e.g. 73 or 181:float32)")
    g.add_argument("--y", metavar="BYTE[:TYPE]", help="Y coordinate byte")
    g.add_argument("--scalar", type=int, help="coordinate scalar to use (SEG-Y convention: -100 = divide by 100)")
    g.add_argument("--crs", help="coordinate reference system, e.g. EPSG:23031")
    g.add_argument("--format", type=int, dest="format_code", help="sample format code (1=IBM, 5=IEEE, ...)")
    g.add_argument("--dt", type=float, help="sample interval in milliseconds (if the file lacks it)")
    g.add_argument("--swap-xy", action="store_true", default=None, help="X and Y are swapped in the file")
    g.add_argument("--loadsheet", help="apply the decisions saved in a load sheet JSON (from a confirmed file)")


def overrides_from_args(a) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if getattr(a, "loadsheet", None):
        with open(a.loadsheet) as f:
            ls = json.load(f)
        o.update(ls.get("overrides", ls))
    for k in ("inline", "crossline", "cdp", "x", "y"):
        r = _ref(getattr(a, k, None))
        if r:
            o[k] = r.to_dict()
    if a.scalar is not None:
        o["scalar"] = a.scalar
    if a.crs:
        o["epsg"] = _epsg(a.crs)
    if a.format_code:
        o["format_code"] = a.format_code
    if a.dt:
        o["dt_us"] = a.dt * 1000
    if a.swap_xy:
        o["swap_xy"] = True
    return o


# ------------------------------------------------------------------ printing
def print_report(rep, verbose: bool = False) -> None:
    print()
    print(f"{_status(rep.status)} {C.bold(rep.file_name)}  {C.dim(f'{rep.size / 1e6:,.1f} MB')}")
    print(f"  {rep.summary}")
    if rep.status_reason:
        print(f"  {C.dim(rep.status_reason)}")
    print()
    from .exports import load_instructions

    li = load_instructions(rep, fixed=False)
    print(C.bold("  How to load the ORIGINAL file"))
    for k, v in li["rows"]:
        v = "-" if v is None else str(v)
        flag = C.yellow(v) if "OVERRIDE" in v or "swapped" in v else v
        print(f"    {k:<26} {flag}")
    G = rep.geometry
    if G.get("inline_spacing"):
        print(f"    {'Bin size':<26} {G['inline_spacing']:.2f} x {G['crossline_spacing']:.2f}  "
              f"(inline azimuth {G['inline_azimuth']:.1f} deg)")
    elif G.get("trace_spacing"):
        print(f"    {'Trace spacing':<26} {G['trace_spacing']:.2f}")
    if G.get("corners"):
        print(C.bold("\n  Survey corners (for survey set-up)"))
        print(C.dim(f"    {'inline':>8} {'crossline':>10} {'X':>14} {'Y':>14}"))
        for c in G["corners"]:
            print(f"    {c['inline']:>8} {c['crossline']:>10} {c['x']:>14.2f} {c['y']:>14.2f}")
    if G.get("centre_lonlat"):
        lon, lat = G["centre_lonlat"]
        where = "on land" if G.get("centre_on_land") else "at sea"
        print(f"\n  Centre {lat:.4f}{'N' if lat >= 0 else 'S'} {abs(lon):.4f}{'E' if lon >= 0 else 'W'} ({where})")
    if rep.issues:
        print(C.bold("\n  Issues"))
        for i in rep.issues:
            col = {"error": C.red, "warning": C.yellow, "info": C.dim}[i.severity]
            tag = C.green(" [fixed in output]") if i.auto_fixed else ""
            print(f"    {col(i.severity.upper()[:4])}  {i.title}{tag}")
            if verbose:
                print(f"          {C.dim(i.detail)}")
                if i.fix:
                    print(f"          {C.dim('-> ' + i.fix)}")
    if verbose:
        print(C.bold("\n  Evidence"))
        for f in rep.findings:
            print(f"    {f.label}: {f.display}  {C.dim('[' + f.confidence + ', ' + f.source + ']')}")
            for e in f.evidence[:4]:
                print(f"        {C.dim('- ' + e)}")


def _write_outputs(rep, out_dir: str, fixlog=None, html: bool = True, stem: str | None = None) -> dict[str, str]:
    from .exports import geojson, loadsheet, write_json

    os.makedirs(out_dir, exist_ok=True)
    stem = stem or os.path.splitext(rep.file_name)[0]
    paths = {}
    paths["loadsheet"] = os.path.join(out_dir, f"{stem}.loadsheet.json")
    write_json(paths["loadsheet"], loadsheet(rep, fixlog))
    paths["report_json"] = os.path.join(out_dir, f"{stem}.report.json")
    write_json(paths["report_json"], rep.to_dict())
    gj = geojson(rep)
    if gj["features"]:
        paths["geojson"] = os.path.join(out_dir, f"{stem}.footprint.geojson")
        write_json(paths["geojson"], gj)
    if html:
        from .report_html import render_report

        paths["html"] = os.path.join(out_dir, f"{stem}.report.html")
        with open(paths["html"], "w") as f:
            f.write(render_report(rep, fixlog))
    return paths


def _progress_printer(enabled: bool):
    if not enabled or not sys.stderr.isatty():
        return None
    last = {"s": ""}

    def p(stage: str, frac: float):
        msg = f"\r  {stage} {frac * 100:5.1f}%"
        if msg != last["s"]:
            sys.stderr.write(msg + " " * 10)
            sys.stderr.flush()
            last["s"] = msg
        if frac >= 1.0 and stage == "Done":
            sys.stderr.write("\r" + " " * 60 + "\r")

    return p


def _expand(paths: list[str], recursive: bool = True) -> list[str]:
    from .inventory import find_segy

    out = []
    for p in paths:
        if os.path.isdir(p):
            out += find_segy(p, recursive)
        else:
            out.append(p)
    return out


# ------------------------------------------------------------------ commands
def cmd_scan(a) -> int:
    from .analyze import analyze

    ov = overrides_from_args(a)
    worst = 0
    files = _expand(a.files)
    if not files:
        print("No SEG-Y files found.", file=sys.stderr)
        return 2
    results = []
    for path in files:
        if not os.path.exists(path):
            print(C.red(f"Not found: {path}"), file=sys.stderr)
            worst = 2
            continue
        rep = analyze(path, overrides=ov, progress=_progress_printer(not a.json), full_scan=not a.quick)
        results.append(rep)
        if a.json:
            continue
        print_report(rep, verbose=a.verbose)
        if not a.no_files:
            out_dir = a.out or os.path.join(os.getcwd(), "segwise_out")
            paths = _write_outputs(rep, out_dir, html=not a.no_html)
            print(C.dim(f"\n  Report: {paths.get('html', paths['report_json'])}"))
            print(C.dim(f"  Load sheet: {paths['loadsheet']}"))
        worst = max(worst, STATUS_EXIT.get(rep.status, 2))
    if a.json:
        out = [r.to_dict(include_plot=False) for r in results]
        print(json.dumps(out[0] if len(out) == 1 else out, indent=2))
        worst = max([STATUS_EXIT.get(r.status, 2) for r in results] or [2])
    return worst


def cmd_fix(a) -> int:
    from .analyze import analyze
    from .writer import WriteError, write_fixed

    ov = overrides_from_args(a)
    files = _expand(a.files)
    multi = len(files) > 1 or (a.output and os.path.isdir(a.output))
    if multi and a.output and not os.path.isdir(a.output):
        os.makedirs(a.output, exist_ok=True)
    worst = 0
    for path in files:
        rep = analyze(path, overrides=ov, progress=_progress_printer(True))
        if rep.status == "unusable":
            print_report(rep)
            print(C.red("  Cannot write a fixed file: the input is unusable."))
            worst = 2
            continue
        blocking = [i for i in rep.issues if i.code in ("CRS_AMBIGUOUS", "CRS_UNKNOWN") and not a.force]
        stem = os.path.splitext(rep.file_name)[0]
        if multi:
            out = os.path.join(a.output or os.path.dirname(path), f"{stem}_std.sgy")
        else:
            out = a.output or os.path.join(os.path.dirname(os.path.abspath(path)), f"{stem}_std.sgy")
        nav = None
        if getattr(a, "nav", None):
            from .nav import match, read_nav

            m = match(rep, read_nav(a.nav), line=a.nav_line, key_byte=a.nav_key, nav_epsg=_epsg(a.nav_epsg),
                      to_epsg=_epsg(a.to_epsg))
            nav = m.as_writer_nav()
            print(C.bold("\n  Navigation merge"))
            for n_ in m.notes:
                print(f"    - {n_}")
        print_report(rep, verbose=a.verbose)
        if blocking and nav is None:
            print(C.yellow(f"\n  Note: {blocking[0].title}. The fixed file will record the CRS as not confirmed; "
                           "pass --crs EPSG:xxxx to set it."))
        try:
            log = write_fixed(rep, out, ieee=a.ieee, nav=nav, progress=_progress_printer(True), overwrite=a.overwrite)
        except WriteError as e:
            print(C.red(f"\n  {e}"))
            worst = 2
            continue
        if nav is not None:
            # re-analyse the output so the report and load sheet describe the merged file
            rep2 = analyze(out, overrides={"epsg": m.to_epsg} if m.to_epsg else None)
            rep2.notes = m.notes + rep2.notes
            rep_for_files = rep2
        else:
            rep_for_files = rep
        print(C.bold("\n  Fixed file"))
        print(f"    {out}")
        for x in log.actions:
            print(f"    {C.green('+')} {x}")
        for w in log.warnings:
            print(f"    {C.yellow('!')} {w}")
        print(C.bold("\n  Verification"))
        for c in log.verification.get("checks", []):
            print(f"    {C.green('PASS') if c['ok'] else C.red('FAIL')}  {c['name']}  {C.dim(c['detail'])}")
        out_dir = a.reports or os.path.dirname(out)
        paths = _write_outputs(rep_for_files, out_dir, fixlog=log, html=not a.no_html,
                               stem=os.path.splitext(os.path.basename(out))[0])
        print(C.dim(f"\n  Report: {paths.get('html')}\n  Load sheet: {paths['loadsheet']}"))
        if not log.verification.get("passed", False):
            worst = 2
        else:
            worst = max(worst, 0 if rep.status in ("ready", "fixable") else 1)
    return worst


def cmd_inventory(a) -> int:
    from .inventory import run_inventory

    res = run_inventory(a.directory, a.output, jobs=a.jobs, recursive=not a.no_recursive, overrides=overrides_from_args(a),
                        progress=True)
    print(f"\n{C.bold('Inventory')} of {res['n_files']} files -> {res['output']}")
    for k, v in res["by_status"].items():
        print(f"  {_status(k)} {v}")
    print(f"  {res['n_groups']} distinct header layouts (files in a group can share one confirmed load sheet)")
    print(C.dim(f"\n  Open {os.path.join(res['output'], 'index.html')}"))
    return 0 if res["by_status"].get("unusable", 0) == 0 else 1


def cmd_headers(a) -> int:
    import numpy as np

    from .fields import TRACE_FIELD_BY_BYTE
    from .reader import TraceReader, probe

    lay = probe(a.file)
    r = TraceReader(lay)
    specs = []
    for tok in (a.bytes or "1,5,9,13,17,21,37,71,73,77,181,185,189,193,197").split(","):
        ref = _ref(tok.strip(), "int16" if (TRACE_FIELD_BY_BYTE.get(int(tok.split(':')[0])) and TRACE_FIELD_BY_BYTE[int(tok.split(':')[0])].size == 2 and ':' not in tok) else "int32")
        specs.append(ref)
    n = lay.n_traces
    idx = np.arange(min(n, a.n)) if not a.all else np.arange(n)
    cols = [r.field(idx, s.byte, s.type) for s in specs]
    names = [f"{s.byte}" + ("" if s.type == "int32" else f":{s.type}") for s in specs]
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trace"] + [f"{n_} {TRACE_FIELD_BY_BYTE[s.byte].name if s.byte in TRACE_FIELD_BY_BYTE else ''}".strip()
                                    for n_, s in zip(names, specs)])
            for i, t in enumerate(idx):
                w.writerow([int(t)] + [c[i].item() for c in cols])
        print(f"Wrote {len(idx)} rows to {a.csv}")
        return 0
    print(C.dim("trace ") + " ".join(f"{n_:>11}" for n_ in names))
    for i, t in enumerate(idx):
        print(f"{int(t):>5} " + " ".join(f"{c[i].item():>11}" for c in cols))
    return 0


def cmd_nav(a) -> int:
    a.files = [a.file]
    return cmd_fix(a)


def cmd_ui(a) -> int:
    try:
        from .web.app import serve
    except ImportError as e:
        print(f"The web app needs extra packages: pip install 'segwise[ui]' ({e})", file=sys.stderr)
        return 2
    serve(host=a.host, port=a.port, root=a.root, open_browser=not a.no_browser)
    return 0


def cmd_mcp(a) -> int:
    try:
        from .mcp_server import main as mcp_main
    except ImportError as e:
        print(f"The MCP server needs: pip install 'segwise[mcp]' ({e})", file=sys.stderr)
        return 2
    mcp_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="segwise", description="Forensic SEG-Y loader: find the real geometry, coordinates "
                                "and CRS, prove it, and write a clean file that loads first time.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--version", action="version", version=f"segwise {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="analyse files and write a report + load sheet")
    s.add_argument("files", nargs="+", help="SEG-Y files or directories")
    s.add_argument("-o", "--out", help="directory for reports (default ./segwise_out)")
    s.add_argument("-v", "--verbose", action="store_true", help="show issue details and evidence")
    s.add_argument("--json", action="store_true", help="print the full report as JSON (no files written)")
    s.add_argument("--no-files", action="store_true", help="print only; write no report files")
    s.add_argument("--no-html", action="store_true", help="skip the HTML report")
    s.add_argument("--quick", action="store_true", help="sample headers instead of reading every trace header")
    add_override_args(s)
    s.set_defaults(func=cmd_scan)

    f = sub.add_parser("fix", help="write a standardised, verified copy (never modifies the original)")
    f.add_argument("files", nargs="+")
    f.add_argument("-o", "--output", help="output file (single input) or directory (several inputs)")
    f.add_argument("--reports", help="directory for the report and load sheet (default: next to the output)")
    f.add_argument("--ieee", action="store_true", help="convert samples to IEEE float (format 5)")
    f.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    f.add_argument("--force", action="store_true", help="write even if the CRS is unresolved")
    f.add_argument("-v", "--verbose", action="store_true")
    f.add_argument("--no-html", action="store_true")
    f.add_argument("--nav", help="navigation file to merge (P1/90 or CSV/whitespace table)")
    f.add_argument("--nav-line", help="line name in the navigation file (default: matched automatically)")
    f.add_argument("--nav-key", type=int, help="trace-header byte the navigation key matches (default: automatic)")
    f.add_argument("--nav-epsg", help="CRS of the navigation positions (default WGS 84 for lat/lon)")
    f.add_argument("--to-epsg", help="CRS to write coordinates in (default: UTM zone of the line)")
    add_override_args(f)
    f.set_defaults(func=cmd_fix)

    n = sub.add_parser("nav", help="merge navigation into a file that lacks coordinates (alias of fix --nav)")
    n.add_argument("file")
    n.add_argument("--nav", required=True)
    n.add_argument("-o", "--output")
    n.add_argument("--reports")
    n.add_argument("--nav-line")
    n.add_argument("--nav-key", type=int)
    n.add_argument("--nav-epsg")
    n.add_argument("--to-epsg")
    n.add_argument("--ieee", action="store_true")
    n.add_argument("--overwrite", action="store_true")
    n.add_argument("--force", action="store_true")
    n.add_argument("-v", "--verbose", action="store_true")
    n.add_argument("--no-html", action="store_true")
    add_override_args(n)
    n.set_defaults(func=cmd_nav)

    i = sub.add_parser("inventory", help="batch-scan a directory (a data delivery)")
    i.add_argument("directory")
    i.add_argument("-o", "--output", default="segwise_inventory", help="output directory")
    i.add_argument("-j", "--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    i.add_argument("--no-recursive", action="store_true")
    add_override_args(i)
    i.set_defaults(func=cmd_inventory)

    h = sub.add_parser("headers", help="print or export trace-header values")
    h.add_argument("file")
    h.add_argument("--bytes", help="comma-separated byte list, e.g. 9,21,73:int32,71:int16")
    h.add_argument("-n", type=int, default=20, help="number of traces (default 20)")
    h.add_argument("--all", action="store_true", help="every trace")
    h.add_argument("--csv", help="write CSV instead of printing")
    h.set_defaults(func=cmd_headers)

    u = sub.add_parser("ui", help="start the local web app")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--root", default=os.getcwd(), help="folder the app may browse (default: current directory)")
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(func=cmd_ui)

    m = sub.add_parser("mcp", help="run as an MCP server so AI agents can call segwise as tools")
    m.set_defaults(func=cmd_mcp)
    return p


def _selftest() -> int:
    import matplotlib
    from pyproj import CRS

    from . import plots  # noqa: F401  (selects the off-screen plotting backend)
    from .report_html import css, env

    checks = {"crs database": CRS.from_epsg(23031).name == "ED50 / UTM zone 31N",
              "plotting": matplotlib.get_backend().lower() == "agg",
              "templates": bool(css()) and env().get_template("web_home.html") is not None}
    try:
        import tkinter  # noqa: F401
        from tkinter import filedialog  # noqa: F401

        checks["file dialog"] = True
    except Exception:
        checks["file dialog"] = sys.platform == "darwin"  # macOS uses the Finder dialog instead
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        checks["web app"] = True
    except Exception:
        checks["web app"] = False
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    return 0 if all(checks.values()) else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_pick"]:  # internal: native file dialog in its own process (see picker.py)
        from .picker import tk_dialog

        tk_dialog(argv[1] if len(argv) > 1 else "file", argv[2] if len(argv) > 2 else None)
        return 0
    if argv[:1] == ["_selftest"]:  # internal: used by the build pipeline to check a packaged app
        return _selftest()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(C.red(f"error: {e}"), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
