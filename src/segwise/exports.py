"""Outputs that plug into people's existing workflows.

* load sheet (JSON): every decision, its evidence and confidence, re-usable as
  overrides for other files from the same delivery
* load instructions for common software (byte locations to type in), for both
  the original file and the fixed file
* code snippets for segyio / segysak (Python)
* GeoJSON footprint for GIS (QGIS, ArcGIS) and data-management catalogues
* CSV inventory rows for batch work
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import os
from typing import Any

from . import __version__

LOADSHEET_SCHEMA = "segwise.loadsheet/1"


def _bytes(ref: dict | None) -> str:
    if not ref:
        return "-"
    size = 2 if ref.get("type") == "int16" else 4
    return f"{ref['byte']}-{ref['byte'] + size - 1}" + ("" if ref.get("type") == "int32" else f" ({ref['type']})")


def load_instructions(report, fixed: bool = False) -> dict[str, Any]:
    G = report.geometry
    if fixed:
        o = report.recipe["output"]
        il, xl, x, y, cdp = o.get("inline"), o.get("crossline"), o.get("x"), o.get("y"), o.get("cdp") if not G.get("inline") else None
        scal = None
        fmt = o["format"]
        endian = "big"
    else:
        il, xl, x, y, cdp = G.get("inline"), G.get("crossline"), G.get("x"), G.get("y"), G.get("cdp") or G.get("ensemble")
        scal = G.get("scalar_effective")
        fmt = report.samples.get("format")
        endian = report.layout.get("endian")
    crs = (report.crs or {}).get("chosen")
    rows = []
    if il:
        rows += [("Inline number", _bytes(il)), ("Crossline number", _bytes(xl))]
    if cdp:
        rows.append(("CDP / ensemble number", _bytes(cdp)))
    if x:
        rows += [("X coordinate (easting)", _bytes(x)), ("Y coordinate (northing)", _bytes(y))]
        if fixed:
            rows.append(("Coordinate scalar", "bytes 71-72 (apply from header)"))
        elif scal is not None and scal != G.get("scalar_header"):
            rows.append(("Coordinate scalar", f"OVERRIDE: use {scal} (header value {G.get('scalar_header')} is wrong)"))
        else:
            rows.append(("Coordinate scalar", f"bytes 71-72 (apply from header; value {scal})"))
    rows += [("Sample format", f"{fmt}"), ("Byte order", endian),
             ("Samples / interval", f"{report.layout.get('ns')} @ {report.layout.get('dt_us', 0) / 1000:g} ms")]
    if crs:
        word = {"confirmed": "confirmed", "likely": "likely, please confirm"}.get(report.crs.get("status"), "not confirmed")
        rows.append(("Coordinate system", f"EPSG:{crs['epsg']} {crs['name']} ({word})"))
    if G.get("swap_xy") and not fixed:
        rows.append(("Note", "X and Y are swapped in this file. Map 'X' to the northing field or use the fixed file"))
    corners = G.get("corners")
    return {"rows": rows, "corners": corners, "snippets": snippets(report, fixed)}


def snippets(report, fixed: bool = False) -> dict[str, str]:
    G = report.geometry
    path = report.path
    if fixed:
        il, xl, x, y = 189, 193, 181, 185
    else:
        il = (G.get("inline") or {}).get("byte")
        xl = (G.get("crossline") or {}).get("byte")
        x = (G.get("x") or {}).get("byte")
        y = (G.get("y") or {}).get("byte")
    out: dict[str, str] = {}
    fname = "<fixed file>.sgy" if fixed else path
    std_types = fixed or all((G.get(k) or {}).get("type", "int32") == "int32" for k in ("inline", "crossline"))
    if il and xl and G.get("kind") == "3D post-stack":
        lines = ["import segyio", "",
                 f"with segyio.open({fname!r}, iline={il}, xline={xl}) as f:",
                 "    print(f.ilines, f.xlines)",
                 "    cube = segyio.tools.cube(f)"]
        if not std_types:
            lines.insert(0, "# NOTE: the inline/crossline fields are not 4-byte integers; segyio needs the fixed file.")
        out["segyio (Python)"] = "\n".join(lines)
        extra = {"cdp_x": x, "cdp_y": y} if x else {}
        sk = ["import xarray as xr  # pip install segysak", "",
              f"ds = xr.open_dataset({fname!r}, engine='sgy_engine',",
              f"                     dim_byte_fields={{'iline': {il}, 'xline': {xl}}}"
              + (f",\n                     extra_byte_fields={extra})" if extra else ")")]
        if x and not fixed and G.get("scalar_effective") not in (None, G.get("scalar_header")):
            sk.append(f"# coordinates: apply scalar {G.get('scalar_effective')} yourself; the header value is wrong")
        else:
            sk.append("ds.segysak.scale_coords()  # applies the coordinate scalar from byte 71")
        out["segysak / xarray (Python)"] = "\n".join(sk)
    else:
        out["segyio (Python)"] = "\n".join(["import segyio", "",
                                            f"with segyio.open({fname!r}, ignore_geometry=True) as f:",
                                            "    data = segyio.tools.collect(f.trace[:])",
                                            f"    cdp = f.attributes({(G.get('cdp') or G.get('ensemble') or {'byte': 21})['byte'] if not fixed else 21})[:]"])
    return out


def loadsheet(report, fixlog=None) -> dict[str, Any]:
    G = report.geometry
    d = {
        "schema": LOADSHEET_SCHEMA,
        "generated_by": f"segwise {__version__}",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {"path": report.path, "file_name": report.file_name, "size_bytes": report.size,
                   "fingerprint": report.fingerprint},
        "status": report.status, "status_reason": report.status_reason, "summary": report.summary,
        "layout": {k: report.layout.get(k) for k in ("endian", "text_encoding", "revision", "ns", "dt_us", "n_traces",
                                                      "data_offset", "variable_length", "is_su")}
        | {"sample_format": report.samples.get("format"), "declared_format": report.samples.get("declared_format")},
        "geometry": {k: G.get(k) for k in (
            "kind", "inline", "crossline", "cdp", "ensemble", "sp", "offset", "x", "y", "scalar_header", "scalar_effective",
            "swap_xy", "coord_type", "coordinate_units_code", "il_min", "il_max", "il_inc", "xl_min", "xl_max", "xl_inc",
            "n_inlines", "n_crosslines", "fill", "key_min", "key_max", "fold", "inline_spacing", "crossline_spacing",
            "inline_azimuth", "trace_spacing", "line_length", "corners", "centre_lonlat") if G.get(k) is not None},
        "crs": report.crs,
        "decisions": [{"key": f.key, "label": f.label, "value": f.value, "display": f.display, "confidence": f.confidence,
                       "source": f.source, "evidence": f.evidence} for f in report.findings],
        "issues": [i.to_dict() for i in report.issues],
        "load_original_file": {r[0]: r[1] for r in load_instructions(report, False)["rows"]},
        "overrides": reusable_overrides(report),
    }
    if fixlog is not None:
        d["fixed_file"] = {"path": fixlog.output, "actions": fixlog.actions, "warnings": fixlog.warnings,
                           "verification": fixlog.verification, "output_scalar": fixlog.output_scalar,
                           "output_format": fixlog.output_format,
                           "load": {r[0]: r[1] for r in load_instructions(report, True)["rows"]}}
    return _clean(d)


def reusable_overrides(report) -> dict[str, Any]:
    """The decisions, in the form `analyze(..., overrides=...)` accepts - so a confirmed layout can be applied to
    every other file of the same delivery."""
    G = report.geometry
    o: dict[str, Any] = {}
    for k in ("inline", "crossline", "x", "y"):
        if G.get(k):
            o[k] = G[k]
    if G.get("cdp") and not G.get("inline"):
        o["cdp"] = G["cdp"]
    if G.get("scalar_effective") is not None and G.get("scalar_effective") != G.get("scalar_header"):
        o["scalar"] = G["scalar_effective"]
    if (report.crs or {}).get("chosen"):
        o["epsg"] = report.crs["chosen"]["epsg"]
    if report.samples.get("format") != report.samples.get("declared_format"):
        o["format_code"] = report.samples.get("format")
    if G.get("swap_xy"):
        o["swap_xy"] = True
    return o


def _clean(o):
    from .analyze import _jsonable

    return _jsonable(o)


def geojson(report) -> dict[str, Any]:
    feats = []
    props = {"file": report.file_name, "status": report.status, "kind": report.geometry.get("kind"),
             "summary": report.summary, "fingerprint": report.fingerprint,
             "epsg": ((report.crs or {}).get("chosen") or {}).get("epsg"), "crs_status": (report.crs or {}).get("status")}
    G = report.geometry
    for k in ("il_min", "il_max", "xl_min", "xl_max", "key_min", "key_max", "n_traces"):
        if G.get(k) is not None:
            props[k] = G[k]
    ol = report.plot.get("outline_lonlat")
    if ol:
        ol = [[float(a), float(b)] for a, b in ol]
        if G.get("kind", "").startswith("3D") or (len(ol) > 3 and ol[0] == ol[-1]):
            geom = {"type": "Polygon", "coordinates": [ol]}
        else:
            geom = {"type": "LineString", "coordinates": ol}
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats,
            "note": "Coordinates are WGS 84 longitude/latitude, converted from the file's CRS" if feats else
            "No footprint: the file has no usable coordinates or no CRS was determined."}


INVENTORY_COLUMNS = ["file_name", "path", "status", "status_reason", "kind", "n_traces", "ns", "dt_ms", "format",
                     "inline_bytes", "crossline_bytes", "x_bytes", "y_bytes", "scalar", "il_range", "xl_range", "key_range",
                     "bin_m", "epsg", "crs_status", "centre_lon", "centre_lat", "errors", "warnings", "layout_group",
                     "size_mb", "fingerprint"]


def inventory_row(report) -> dict[str, Any]:
    G = report.geometry
    c = report.counts
    ch = (report.crs or {}).get("chosen") or {}
    cl = G.get("centre_lonlat") or [None, None]
    return {
        "file_name": report.file_name, "path": report.path, "status": report.status, "status_reason": report.status_reason,
        "kind": G.get("kind"), "n_traces": report.layout.get("n_traces"), "ns": report.layout.get("ns"),
        "dt_ms": (report.layout.get("dt_us") or 0) / 1000, "format": report.samples.get("format"),
        "inline_bytes": _bytes(G.get("inline")) if G.get("inline") else "",
        "crossline_bytes": _bytes(G.get("crossline")) if G.get("crossline") else "",
        "x_bytes": _bytes(G.get("x")) if G.get("x") else "", "y_bytes": _bytes(G.get("y")) if G.get("y") else "",
        "scalar": G.get("scalar_effective"),
        "il_range": f"{G['il_min']}-{G['il_max']}" if "il_min" in G else "",
        "xl_range": f"{G['xl_min']}-{G['xl_max']}" if "xl_min" in G else "",
        "key_range": f"{G['key_min']}-{G['key_max']}" if "key_min" in G else "",
        "bin_m": (f"{G['inline_spacing']:.2f}x{G['crossline_spacing']:.2f}" if G.get("inline_spacing") else
                  (f"{G['trace_spacing']:.2f}" if G.get("trace_spacing") else "")),
        "epsg": ch.get("epsg"), "crs_status": (report.crs or {}).get("status"),
        "centre_lon": cl[0], "centre_lat": cl[1], "errors": c.get("error", 0), "warnings": c.get("warning", 0),
        "layout_group": report.layout_key, "size_mb": round(report.size / 1e6, 2), "fingerprint": report.fingerprint,
    }


def inventory_csv(reports) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=INVENTORY_COLUMNS)
    w.writeheader()
    for r in reports:
        w.writerow(inventory_row(r))
    return buf.getvalue()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
