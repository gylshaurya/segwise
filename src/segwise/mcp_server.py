"""MCP server: lets AI assistants and agents that support the Model Context Protocol use segwise as tools.

    segwise mcp            # stdio transport

The tools are deterministic and evidence-based; an agent gets the same verdicts,
evidence and verification a person sees, and can never write over an original file.
"""
from __future__ import annotations

import os
from typing import Any

try:  # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # v1
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = (
    "segwise inspects SEG-Y seismic files. Use scan_segy first: it returns the verdict (ready / fixable / "
    "needs_review / unusable), the byte locations of inline, crossline and coordinates, the CRS and every issue with "
    "evidence. Treat 'needs_review' as requiring a human or a stated fact (often the CRS: pass epsg in overrides only "
    "if the user or a document states it). fix_segy writes a standardised copy and returns verification results; it "
    "never overwrites the input. All processing is local."
)

server = _Server(name="segwise", instructions=INSTRUCTIONS)


def _summary(rep) -> dict[str, Any]:
    from .exports import load_instructions, reusable_overrides

    G = rep.geometry
    keep = ("kind", "inline", "crossline", "cdp", "ensemble", "x", "y", "scalar_effective", "swap_xy", "il_min", "il_max",
            "xl_min", "xl_max", "key_min", "key_max", "inline_spacing", "crossline_spacing", "inline_azimuth",
            "trace_spacing", "corners", "centre_lonlat", "centre_on_land", "coord_type")
    return {
        "file": rep.path, "status": rep.status, "status_reason": rep.status_reason, "summary": rep.summary,
        "layout": {k: rep.layout.get(k) for k in ("endian", "ns", "dt_us", "n_traces", "revision", "text_encoding")},
        "sample_format": rep.samples.get("format"),
        "geometry": {k: G.get(k) for k in keep if G.get(k) is not None},
        "crs": {"status": rep.crs.get("status"), "chosen": rep.crs.get("chosen"),
                "candidates": [{"epsg": c["epsg"], "name": c["name"], "reasons": c["reasons"]} for c in rep.crs.get("candidates", [])[:5]],
                "notes": rep.crs.get("notes")},
        "issues": [{"code": i.code, "severity": i.severity, "title": i.title, "detail": i.detail, "auto_fixed": i.auto_fixed}
                   for i in rep.issues],
        "evidence": {f.key: {"value": f.display, "confidence": f.confidence, "evidence": f.evidence[:4]} for f in rep.findings},
        "load_instructions": dict(load_instructions(rep, False)["rows"]) if rep.status != "unusable" else {},
        "reusable_overrides": reusable_overrides(rep) if rep.status != "unusable" else {},
    }


def _jsonable(o):
    from .analyze import _jsonable as j

    return j(o)


@server.tool(description="Analyse one SEG-Y file and return the verdict, byte locations, CRS, issues and evidence. "
                         "overrides (optional) may set inline/crossline/x/y ({'byte': 9, 'type': 'int32'}), scalar, "
                         "epsg, format_code, swap_xy, dt_us.")
def scan_segy(path: str, overrides: dict | None = None) -> dict:
    from .analyze import analyze

    rep = analyze(os.path.expanduser(path), overrides=overrides)
    return _jsonable(_summary(rep))


@server.tool(description="Write a standardised, verified SEG-Y rev 1 copy (keys at 189/193/181/185). Never overwrites "
                         "the input. Returns the actions taken and verification checks, plus a load sheet path.")
def fix_segy(path: str, output: str, overrides: dict | None = None, convert_to_ieee: bool = False) -> dict:
    from .analyze import analyze
    from .cli import _write_outputs
    from .writer import write_fixed

    rep = analyze(os.path.expanduser(path), overrides=overrides)
    if rep.status == "unusable":
        return {"error": "Input is unusable", "status_reason": rep.status_reason}
    out = os.path.expanduser(output)
    log = write_fixed(rep, out, ieee=convert_to_ieee)
    paths = _write_outputs(rep, os.path.dirname(os.path.abspath(out)), fixlog=log, stem=os.path.splitext(os.path.basename(out))[0])
    return _jsonable({"output": log.output, "actions": log.actions, "warnings": log.warnings,
                      "verification": log.verification, "loadsheet": paths["loadsheet"], "report": paths.get("html")})


@server.tool(description="Scan every SEG-Y file under a directory. Writes inventory.csv, inventory.geojson, index.html "
                         "and per-file reports to output_dir; returns counts by status and layout groups.")
def inventory_segy(directory: str, output_dir: str, overrides: dict | None = None) -> dict:
    from .inventory import run_inventory

    res = run_inventory(os.path.expanduser(directory), os.path.expanduser(output_dir), jobs=2, overrides=overrides)
    res.pop("failures", None)
    return _jsonable(res)


@server.tool(description="Read raw trace-header values for given 1-based byte positions (e.g. '9,21,73:int32,71:int16') "
                         "for traces start..start+count.")
def header_values(path: str, bytes: str = "1,5,9,21,71,73,77,181,185,189,193", start: int = 0, count: int = 20) -> dict:
    import numpy as np

    from .fields import TRACE_FIELD_BY_BYTE
    from .reader import TraceReader, probe

    lay = probe(os.path.expanduser(path))
    r = TraceReader(lay)
    idx = np.arange(max(0, start), min(lay.n_traces, start + min(count, 1000)))
    cols = {}
    for tok in bytes.split(","):
        b, _, t = tok.strip().partition(":")
        f = TRACE_FIELD_BY_BYTE.get(int(b))
        t = t or ("int16" if f and f.size == 2 else "int32")
        cols[f"{b}:{t}" + (f" {f.name}" if f else "")] = r.field(idx, int(b), t).tolist()
    return {"traces": idx.tolist(), "columns": cols}


@server.tool(description="Merge navigation (UKOOA P1/90 or a CSV/whitespace table) into a SEG-Y file that lacks "
                         "coordinates, writing a new file. The nav line and key column are matched from the data.")
def merge_navigation(path: str, nav_path: str, output: str, line: str | None = None, to_epsg: int | None = None,
                     nav_epsg: int | None = None) -> dict:
    from .analyze import analyze
    from .nav import match, read_nav
    from .writer import write_fixed

    rep = analyze(os.path.expanduser(path))
    m = match(rep, read_nav(os.path.expanduser(nav_path)), line=line, to_epsg=to_epsg, nav_epsg=nav_epsg)
    log = write_fixed(rep, os.path.expanduser(output), nav=m.as_writer_nav())
    return _jsonable({"match": m.summary(), "output": log.output, "verification": log.verification})


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
