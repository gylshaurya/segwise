"""Run the whole forensic analysis of one file and assemble a Report."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from . import __version__
from . import crs as crsmod
from . import geometry as geo
from .amplitude import amplitude_stats, verify_format
from .fields import BIN_LABELS, SORTING_CODES, TRACE_FIELD_BY_BYTE
from .formats import FORMATS, format_name
from .model import FieldRef, Finding, Issue, sort_issues
from .reader import Layout, ProbeError, TraceReader, probe, sample_indices
from .textheader import TextInfo, parse as parse_text

STD_BINS_M = [3.125, 5, 6.25, 10, 12.5, 15, 18.75, 20, 25, 30, 37.5, 40, 50, 60, 62.5, 75, 100]
STD_BINS_FT = [27.5, 41.25, 55, 82.5, 110, 165, 220]


# ------------------------------------------------------------------ overrides
@dataclass
class Overrides:
    """Decisions supplied by a person (or a saved load sheet). Each one wins over inference."""

    inline: FieldRef | None = None
    crossline: FieldRef | None = None
    cdp: FieldRef | None = None
    x: FieldRef | None = None
    y: FieldRef | None = None
    scalar: int | None = None  # SEG-Y convention: >0 multiply, <0 divide
    epsg: int | None = None
    format_code: int | None = None
    swap_xy: bool | None = None
    dt_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in asdict(self).items():
            if v is not None:
                out[k] = v
        return out

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "Overrides":
        d = d or {}
        o = Overrides()
        for k in ("inline", "crossline", "cdp", "x", "y"):
            if d.get(k):
                setattr(o, k, d[k] if isinstance(d[k], FieldRef) else FieldRef.from_dict(d[k]))
        for k in ("scalar", "epsg", "format_code"):
            if d.get(k) not in (None, ""):
                setattr(o, k, int(d[k]))
        if d.get("swap_xy") is not None:
            o.swap_xy = bool(d["swap_xy"])
        if d.get("dt_us") not in (None, ""):
            o.dt_us = float(d["dt_us"])
        return o

    def any(self) -> bool:
        return bool(self.to_dict())


def scalar_factor(s: int) -> float:
    if s > 0:
        return float(s)
    if s < 0:
        return 1.0 / abs(s)
    return 1.0


def factor_to_scalar(f: float) -> int:
    if f >= 1:
        return int(round(f))
    return -int(round(1 / f))


def _near_std(b: float) -> tuple[bool, float | None, str | None]:
    for sb in STD_BINS_M:
        if abs(b - sb) / sb < 0.015:
            return True, sb, "m"
    for sb in STD_BINS_FT:
        if abs(b - sb) / sb < 0.015:
            return True, sb, "ft"
    return False, None, None


# ------------------------------------------------------------------ report
@dataclass
class Report:
    path: str
    file_name: str
    size: int
    tool_version: str = __version__
    scanned_at: str = ""
    fingerprint: str = ""
    status: str = "unusable"  # ready | fixable | needs_review | unusable
    status_reason: str = ""
    summary: str = ""
    layout: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    crs: dict[str, Any] = field(default_factory=dict)
    samples: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    header_map: list[dict[str, Any]] = field(default_factory=list)
    text: dict[str, Any] = field(default_factory=dict)
    binary_header: list[dict[str, Any]] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)  # how to load / what the fixed file will contain
    plot: dict[str, Any] = field(default_factory=dict)  # decimated positions etc. for maps
    layout_key: str = ""
    notes: list[str] = field(default_factory=list)

    def finding(self, key: str) -> Finding | None:
        return next((f for f in self.findings if f.key == key), None)

    def to_dict(self, include_plot: bool = True) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("findings", "issues", "plot")}
        d["findings"] = [f.to_dict() for f in self.findings]
        d["issues"] = [i.to_dict() for i in self.issues]
        if include_plot:
            d["plot"] = self.plot
        return _jsonable(d)

    def to_json(self, include_plot: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_plot), indent=indent)

    @property
    def counts(self) -> dict[str, int]:
        c = {"error": 0, "warning": 0, "info": 0}
        for i in self.issues:
            c[i.severity] = c.get(i.severity, 0) + 1
        return c


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, FieldRef):
        return o.to_dict()
    return o


def quick_fingerprint(path: str, size: int) -> str:
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if size > 131072:
            f.seek(size - 65536)
            h.update(f.read(65536))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ main entry
def analyze(path: str, overrides: Overrides | dict | None = None,
            progress: Callable[[str, float], None] | None = None, full_scan: bool = True) -> Report:
    ov = overrides if isinstance(overrides, Overrides) else Overrides.from_dict(overrides)
    prog = progress or (lambda stage, frac: None)
    path = os.path.abspath(path)
    size = os.path.getsize(path)
    rep = Report(path=path, file_name=os.path.basename(path), size=size,
                 scanned_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    rep.overrides = ov.to_dict()
    rep.fingerprint = quick_fingerprint(path, size)
    prog("Reading file structure", 0.02)
    try:
        lay = probe(path)
    except ProbeError as e:
        rep.status, rep.status_reason = "unusable", str(e)
        rep.summary = f"Not a readable SEG-Y file: {e}"
        rep.issues.append(Issue("NOT_SEGY", "error", "Not a readable SEG-Y file", str(e),
                                explain="The file's structure doesn't match SEG-Y or Seismic Unix at all."))
        return rep

    if ov.dt_us:
        lay.dt_us = float(ov.dt_us)
        lay.issues = [i for i in lay.issues if not i.code.startswith("DT_")]
    elif lay.dt_us <= 0:
        lay.issues.append(Issue(
            "DT_UNKNOWN", "error", "Sample interval is not recorded anywhere",
            "Both the binary header and the trace headers have a sample interval of 0.",
            fix="Supply it with --dt (milliseconds), e.g. --dt 4.",
            explain="The file doesn't say how much time passes between samples, so the vertical axis can't be scaled."))
    rep.issues.extend(lay.issues)
    rep.notes.extend(lay.notes)
    _fill_layout(rep, lay)

    # ---------------------------------------------------------------- textual header
    prog("Reading the textual header", 0.08)
    tinfo = parse_text(lay.text_lines)

    if lay.n_traces == 0:
        rep.status, rep.status_reason = "unusable", "The file contains no traces."
        rep.summary = "File headers only, no traces."
        _fill_text(rep, tinfo, None)
        rep.issues = sort_issues(rep.issues)
        return rep

    reader = TraceReader(lay)

    # ---------------------------------------------------------------- sample format
    prog("Checking the sample format", 0.15)
    fv = verify_format(reader, lay)
    fmt = ov.format_code or fv.effective
    if fv.changed and not ov.format_code:
        rep.issues.append(Issue(
            "FORMAT_WRONG", "error", "Declared sample format is wrong",
            f"Binary header declares format {fv.declared} ({format_name(fv.declared)}); the samples are really format "
            f"{fv.effective} ({format_name(fv.effective)}).",
            fix=f"The fixed file declares format {fv.effective}.", auto_fixed=True,
            explain="Each amplitude value is stored as a number in some encoding. The header names the wrong encoding, so "
            "software would decode every amplitude incorrectly. The image may still look like seismic, but the values "
            "are wrong, which ruins amplitude work (AVO, inversion, attributes).",
        ))
    rep.findings.append(Finding(
        "sample_format", "Sample format", fmt, f"{fmt} - {format_name(fmt)}",
        "high" if ov.format_code else fv.confidence, "user" if ov.format_code else "data", fv.evidence,
        [{"format": k, "plausibility": round(v, 3)} for k, v in sorted(fv.scores.items(), key=lambda t: -t[1])]))

    # ---------------------------------------------------------------- header scan & geometry
    prog("Scanning trace headers", 0.25)
    idx, blocks = sample_indices(lay.n_traces)
    heads = reader.headers(idx)
    cands = geo.scan(heads, lay.endian, blocks)
    claims = {}
    for c in tinfo.byte_claims:
        claims.setdefault(c.what, []).append(c.byte)
    if ov.inline or ov.crossline or ov.cdp:
        g = _geometry_from_overrides(ov, cands, blocks)
    else:
        g = geo.infer(cands, blocks, claims)
        if ov.x and ov.y:
            _apply_coord_override(g, ov, cands, blocks)
    elif_xy = ov.x and ov.y
    if (ov.inline or ov.crossline or ov.cdp) and elif_xy:
        _apply_coord_override(g, ov, cands, blocks)

    rep.header_map = geo.describe_header_map(cands, g)

    # ---------------------------------------------------------------- full pass
    prog("Measuring the full survey", 0.4)
    full = _full_pass(reader, lay, g, full_scan, lambda f: prog("Measuring the full survey", 0.4 + 0.3 * f))
    _drop_blank_keys(rep, full)

    # ---------------------------------------------------------------- coordinates, scalar, CRS
    prog("Checking coordinates and CRS", 0.72)
    _coordinates(rep, lay, g, full, tinfo, ov)

    # ---------------------------------------------------------------- amplitudes
    prog("Checking amplitudes", 0.85)
    st = amplitude_stats(reader, lay, fmt)
    _amplitude_issues(rep, st)
    rep.samples = {"declared_format": fv.declared, "format": fmt, "format_name": format_name(fmt), **st.to_dict()}
    rep.plot["rms_idx"] = st.checked_idx[:2000]
    rep.plot["rms"] = st.trace_rms[:2000]

    # ---------------------------------------------------------------- claims vs data
    _check_claims(rep, tinfo, g, full)
    _fill_text(rep, tinfo, g)
    _findings_geometry(rep, g, lay, claims)
    _recipe_and_status(rep, lay, g, fmt, full)
    rep.issues = sort_issues(_dedupe(rep.issues))
    prog("Done", 1.0)
    return rep


# ------------------------------------------------------------------ pieces
def _fill_layout(rep: Report, lay: Layout) -> None:
    b = lay.binary
    rep.layout = {
        "is_su": lay.is_su,
        "endian": "big" if lay.endian == ">" else "little",
        "text_encoding": lay.text_encoding,
        "revision": f"{b.get('rev_major', 0)}.{b.get('rev_minor', 0)}",
        "n_ext_text": lay.n_ext_text,
        "extra_trace_headers": lay.n_extra_trace_headers,
        "declared_format": b.get("format_code"),
        "ns": lay.ns,
        "dt_us": lay.dt_us,
        "record_length_ms": lay.ns * lay.dt_us / 1000.0 if lay.dt_us else None,
        "n_traces": lay.n_traces,
        "trace_bytes": lay.trace_bytes,
        "variable_length": lay.trace_offsets is not None,
        "data_offset": lay.data_offset,
        "trailing_bytes": lay.remainder,
        "sorting": SORTING_CODES.get(b.get("sorting_code", 0), str(b.get("sorting_code"))),
        "measurement_system": {1: "metres", 2: "feet"}.get(b.get("measurement_system"), "not stated"),
    }
    rep.binary_header = [{"name": k, "label": BIN_LABELS.get(k, k), "value": v} for k, v in b.items()]
    rep.findings.append(Finding("endian", "Byte order", rep.layout["endian"], rep.layout["endian"] + "-endian",
                                "high", "data", ["The binary header only makes sense read this way."]))
    rep.findings.append(Finding("ns", "Samples per trace", lay.ns, f"{lay.ns} samples", "high", "data",
                                ["File size divides exactly into traces of this length." if lay.remainder == 0
                                 else "Best-fitting trace length (the file has a partial trace at the end)."]))
    rep.findings.append(Finding("dt", "Sample interval", lay.dt_us, f"{lay.dt_us:g} µs ({lay.dt_us / 1000:g} ms)",
                                "high" if lay.dt_us > 0 else "low", "binary header", []))


def _geometry_from_overrides(ov: Overrides, cands, blocks) -> geo.Geometry:
    def pick(ref: FieldRef | None):
        if ref is None:
            return None
        c = geo.find(cands, ref.byte, ref.type)
        return c

    il, xl, cdp = pick(ov.inline), pick(ov.crossline), pick(ov.cdp)
    if il is not None and xl is not None:
        g = geo.Geometry("3D post-stack", inline=il, crossline=xl, confidence="high")
        g.pair_unique = geo._pair_stats(il.values, xl.values)
        if g.pair_unique < 0.9:
            g.kind = "3D pre-stack"
            g.fold = 1 / max(g.pair_unique, 1e-9)
        g.evidence.append(f"Inline bytes {il.byte} and crossline bytes {xl.byte} were set by the user.")
        if not (ov.x and ov.y):
            cf, _ = geo._best_coord_pair(il.values, xl.values, il.step or 1, xl.step or 1, cands,
                                         {il.byte, xl.byte, *il.aliases, *xl.aliases})
            g.coords = cf
            if cf:
                g.coord_kind = geo._coord_kind(cf.x.byte)
        return g
    if cdp is not None:
        g = geo.Geometry("2D post-stack", cdp=cdp, confidence="high")
        g.evidence.append(f"CDP/trace key bytes {cdp.byte} set by the user.")
        geo._fit_2d(g, cands, blocks, {cdp.byte})
        return g
    return geo.infer(cands, blocks)


def _apply_coord_override(g: geo.Geometry, ov: Overrides, cands, blocks) -> None:
    cx = geo.find(cands, ov.x.byte, ov.x.type)
    cy = geo.find(cands, ov.y.byte, ov.y.type)
    if cx is None or cy is None:
        return
    if g.inline is not None and g.crossline is not None:
        ax, rx = geo._fit_plane(g.inline.values, g.crossline.values, cx.values)
        ay, ry = geo._fit_plane(g.inline.values, g.crossline.values, cy.values)
        b = max(abs(ax[1]), abs(ax[2]), abs(ay[1]), abs(ay[2]), 1e-9)
        g.coords = geo.CoordFit(cx, cy, ax, ay, rx, ry, max(rx, ry) / b, 1.0)
    else:
        g.coords = geo.CoordFit(cx, cy, np.zeros(3), np.zeros(3), 0, 0, 0, 1.0)
    g.coord_kind = geo._coord_kind(cx.byte)
    g.evidence.append(f"Coordinate bytes {cx.byte}/{cy.byte} were set by the user.")


@dataclass
class FullPass:
    il: np.ndarray | None = None
    xl: np.ndarray | None = None
    key: np.ndarray | None = None  # 2D cdp / ensemble
    sp: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    scalar: np.ndarray | None = None
    units: np.ndarray | None = None
    offset: np.ndarray | None = None
    sampled: bool = False
    idx: np.ndarray | None = None


def _full_pass(reader: TraceReader, lay: Layout, g: geo.Geometry, full_scan: bool, prog) -> FullPass:
    specs: list[tuple[str, int, str]] = []

    def add(name, c):
        if c is not None:
            specs.append((name, c.byte, c.type))

    add("il", g.inline)
    add("xl", g.crossline)
    add("key", g.cdp or g.ensemble)
    add("sp", g.sp)
    add("offset", g.offset)
    if g.coords is not None:
        add("x", g.coords.x)
        add("y", g.coords.y)
    specs.append(("scalar", 71, "int16"))
    specs.append(("units", 89, "int16"))
    fp = FullPass()
    n = lay.n_traces
    # Very large files with tiny traces would make a full pass slow; cap at ~60M traces unless asked.
    if full_scan and (n <= 60_000_000):
        vals = reader.fields_chunked([(b, t) for _, b, t in specs], progress=prog)
        fp.idx = np.arange(n)
    else:
        idx, _ = sample_indices(n, budget=200_000, block=2000)
        fp.idx = idx
        fp.sampled = True
        vals = [reader.field(idx, b, t).astype(np.float64 if t == "float32" else np.int64) for _, b, t in specs]
    for (name, _, _), v in zip(specs, vals):
        setattr(fp, name, v)
    return fp


def _drop_blank_keys(rep: Report, fp: FullPass) -> None:
    """Traces whose key fields are all zero (blank headers, tape padding) must not stretch the grid to 0."""
    keys = [a for a in (fp.il, fp.xl) if a is not None] or ([fp.key] if fp.key is not None else [])
    if not keys or fp.idx is None:
        return
    blank = np.ones(keys[0].size, bool)
    for a in keys:
        blank &= a == 0
    if not blank.any() or blank.all() or blank.mean() > 0.05:
        return
    if all(a[~blank].min() <= 0 <= a[~blank].max() for a in keys):
        return  # zero is a genuine line number in this survey
    for name in ("il", "xl", "key", "sp", "x", "y", "scalar", "units", "offset", "idx"):
        a = getattr(fp, name)
        if a is not None:
            setattr(fp, name, a[~blank])
    rep.geometry["blank_traces"] = int(blank.sum())
    rep.issues.append(Issue(
        "BLANK_KEYS", "warning", "Some traces have blank (zero) line numbers",
        f"{int(blank.sum())} traces have zero in every key field (e.g. trace {int(np.flatnonzero(blank)[0]) + 1}), unlike the "
        "rest of the survey.",
        fix="They are left out of the survey geometry and copied unchanged to the fixed file. Check whether they are "
        "padding or dead traces and remove them if so.",
        explain="A few traces carry no position information at all, often tape padding or blank traces. Counting them "
        "would make the survey look enormous and nearly empty."))


def _increment(v: np.ndarray) -> int:
    u = np.unique(v)
    if u.size < 2:
        return 1
    d = np.diff(u)
    vals, cnt = np.unique(d, return_counts=True)
    return int(vals[cnt.argmax()])


def _coordinates(rep: Report, lay: Layout, g: geo.Geometry, fp: FullPass, tinfo: TextInfo, ov: Overrides) -> None:
    G: dict[str, Any] = {"kind": g.kind, "confidence": g.confidence, "evidence": g.evidence,
                         "sampled": fp.sampled, "n_traces": lay.n_traces}
    rep.geometry = G
    if g.inline is not None:
        G["inline"] = g.inline.ref.to_dict()
        G["crossline"] = g.crossline.ref.to_dict()
    if g.cdp is not None:
        G["cdp"] = g.cdp.ref.to_dict()
    if g.ensemble is not None:
        G["ensemble"] = g.ensemble.ref.to_dict()
    if g.sp is not None:
        G["sp"] = g.sp.ref.to_dict()
    if g.offset is not None:
        G["offset"] = g.offset.ref.to_dict()

    # ---- grid statistics -----------------------------------------------------------------
    if fp.il is not None and fp.xl is not None:
        il, xl = fp.il, fp.xl
        ili, xli = _increment(il), _increment(xl)
        G.update(il_min=int(il.min()), il_max=int(il.max()), il_inc=ili, xl_min=int(xl.min()), xl_max=int(xl.max()), xl_inc=xli)
        n_il = (G["il_max"] - G["il_min"]) // max(abs(ili), 1) + 1
        n_xl = (G["xl_max"] - G["xl_min"]) // max(abs(xli), 1) + 1
        pairs = (il.astype(np.int64) - il.min()) * (int(xl.max() - xl.min()) + 1) + (xl - xl.min())
        upairs, counts = np.unique(pairs, return_counts=True)
        G.update(n_inlines=int(n_il), n_crosslines=int(n_xl), n_positions=int(upairs.size),
                 grid_positions=int(n_il * n_xl), fill=round(upairs.size / max(n_il * n_xl, 1), 4),
                 duplicates=int((counts > 1).sum()) if g.kind == "3D post-stack" else 0,
                 fold_max=int(counts.max()), fold_median=float(np.median(counts)))
        if g.kind == "3D post-stack" and (counts > 1).any() and not fp.sampled:
            rep.issues.append(Issue(
                "DUPLICATE_TRACES", "warning", "Some grid positions have more than one trace",
                f"{int((counts > 1).sum())} inline/crossline positions appear more than once (up to {counts.max()} times).",
                fix="The fixed file keeps every trace; your loader will keep the first or last copy. Check whether two "
                "surveys were merged.",
                explain="In a stacked 3D cube each grid cell should have exactly one trace. Duplicates usually mean two "
                "files or two versions were concatenated."))
        if g.kind == "3D post-stack" and G["fill"] < 0.999:
            rep.notes.append(f"The grid is {G['fill'] * 100:.1f}% filled ({G['grid_positions'] - G['n_positions']} empty "
                             "cells), normal for surveys with irregular outlines.")
    if fp.key is not None:
        k = fp.key
        G.update(key_min=int(k.min()), key_max=int(k.max()), key_inc=_increment(k), n_keys=int(np.unique(k).size))
        if g.ensemble is not None:
            G["fold"] = round(k.size / max(np.unique(k).size, 1), 2)
    if fp.sp is not None:
        G.update(sp_min=int(fp.sp.min()), sp_max=int(fp.sp.max()))
    if fp.offset is not None:
        G.update(offset_min=int(fp.offset.min()), offset_max=int(fp.offset.max()))

    # ---- coordinates ---------------------------------------------------------------------
    scal = fp.scalar
    svals, scnt = np.unique(scal, return_counts=True)
    hdr_scalar = int(svals[scnt.argmax()])
    units_code = int(np.bincount(np.clip(fp.units, 0, 9)).argmax()) if fp.units is not None and fp.units.size else 0
    G["coordinate_units_code"] = units_code
    if g.coords is None or fp.x is None:
        G["coords"] = None
        if g.kind != "unknown":
            rep.issues.append(Issue(
                "NO_COORDINATES", "warning" if g.kind not in ("3D post-stack",) else "error",
                "No positions (map coordinates) in the file",
                "No header field behaves like X/Y coordinates (all candidate coordinate fields are zero, constant or "
                "unrelated to the trace positions).",
                fix="Coordinates must come from elsewhere: a navigation file (use 'segwise nav' to merge P1/90 or "
                "CSV navigation by shotpoint/CDP), or the survey corner points for a 3D grid.",
                explain="Without coordinates the data cannot be placed on a map. Legacy data often kept positions in a "
                "separate navigation file."))
        rep.crs = crsmod.CrsResult("none", "not-applicable", None, []).to_dict()
        return

    cx, cy = g.coords.x, g.coords.y
    G["x"] = cx.ref.to_dict()
    G["y"] = cy.ref.to_dict()
    G["coord_kind"] = g.coord_kind
    raw_x = fp.x.astype(np.float64)
    raw_y = fp.y.astype(np.float64)
    zero = (raw_x == 0) & (raw_y == 0)
    if zero.any() and not zero.all():
        rep.issues.append(Issue(
            "ZERO_COORDINATES", "warning", "Some traces have zero coordinates",
            f"{int(zero.sum())} traces ({zero.mean() * 100:.2f}%) have X = Y = 0.",
            fix="For a regular 3D grid the fixed file recomputes their coordinates from the grid; otherwise they are left as-is "
            "and should be repaired from navigation.",
            explain="Some traces are positioned at 'null island' (the origin of the coordinate system) instead of their "
            "real location. They will appear as stray points on a map."))
    if np.unique(scal).size > 1:
        rep.issues.append(Issue(
            "SCALAR_VARIES", "warning", "Coordinate scalar changes between traces",
            f"Bytes 71-72 take the values {', '.join(map(str, svals[:6]))}; most common {hdr_scalar}.",
            fix="Each trace's own scalar is applied when coordinates are rewritten.",
            explain="The 'multiply/divide coordinates by' instruction is different on different traces, which many loaders "
            "ignore."))
    if cx.type == "float32":
        f_hdr = 1.0
        rep.notes.append("Coordinates are stored as IEEE floats (non-standard); the scalar in bytes 71-72 is not applied.")
    else:
        f_hdr = scalar_factor(hdr_scalar)
    if hdr_scalar == 0 and cx.type != "float32":
        rep.issues.append(Issue(
            "SCALAR_ZERO", "info", "Coordinate scalar is zero",
            "Bytes 71-72 are 0, which SEG-Y does not define; it is treated as 1 (no scaling).",
            explain="The 'multiply coordinates by' value is missing; most software assumes 1.", auto_fixed=True))
    sx = raw_x * (np.where(scal > 0, scal, 1) / np.where(scal < 0, -scal, 1) if cx.type != "float32" else 1)
    sy = raw_y * (np.where(scal > 0, scal, 1) / np.where(scal < 0, -scal, 1) if cx.type != "float32" else 1)
    sx = np.where(scal == 0, raw_x, sx) if cx.type != "float32" else raw_x
    sy = np.where(scal == 0, raw_y, sy) if cx.type != "float32" else raw_y

    geo_hint = any(h.kind == "projection" and h.value == "Geographic" for h in tinfo.hints)
    ctype = crsmod.classify(sx[~zero], sy[~zero], units_code, geo_hint)
    G["coord_type"] = ctype

    # bin size in scaled units
    bins = _bin_geometry(g, fp, sx, sy, zero)
    G.update(bins)

    # ---- scalar sanity (projected only) ----------------------------------------------------
    eff = f_hdr
    if ov.scalar is not None:
        eff = scalar_factor(ov.scalar)
    elif ctype == "projected" and bins.get("bin_check_size"):
        b = bins["bin_check_size"]
        ok, sb, unit = _near_std(b)
        med_abs = float(np.median(np.abs(np.r_[sx[~zero], sy[~zero]]))) if (~zero).any() else 0
        plausible_mag = 1_000 <= med_abs <= 20_000_000
        if ok and plausible_mag:
            G["scalar_evidence"] = f"With scalar {hdr_scalar} the bin/trace spacing is {b:.3f} {unit}, a standard survey size."
        else:
            alt = None
            for k in range(-4, 4):
                f = 10.0 ** k
                if abs(np.log10(f) - 0) < 1e-9:
                    f = 1.0
                ok2, sb2, unit2 = _near_std(b / f_hdr * f)
                mag2 = med_abs / f_hdr * f
                if ok2 and 1_000 <= mag2 <= 20_000_000 and abs(f - f_hdr) / f_hdr > 0.01:
                    alt = (f, sb2, unit2)
                    break
            if alt and not plausible_mag:
                f, sb2, unit2 = alt
                eff = f
                rep.issues.append(Issue(
                    "SCALAR_WRONG", "error", "Coordinate scalar is wrong",
                    f"With the header scalar ({hdr_scalar}) coordinates are around {med_abs:,.0f} and the spacing is {b:.4g} - "
                    f"implausible. With scalar {factor_to_scalar(f)} the spacing is exactly {sb2:g} {unit2} and coordinates "
                    f"are around {med_abs / f_hdr * f:,.0f}.",
                    fix=f"The fixed file writes scalar {factor_to_scalar(f)}.", auto_fixed=True,
                    explain="SEG-Y stores coordinates as whole numbers plus a 'divide by' value so that decimals survive. "
                    "This file has the wrong divisor, so the survey would load 10x, 100x or 1000x too big or too small - "
                    "possibly on another continent."))
            elif alt:
                f, sb2, unit2 = alt
                rep.issues.append(Issue(
                    "SCALAR_SUSPECT", "warning", "Coordinate scalar may be wrong",
                    f"With scalar {hdr_scalar} the spacing is {b:.4g}, not a standard bin size; scalar {factor_to_scalar(f)} "
                    f"would give exactly {sb2:g} {unit2}. Not changed automatically because the coordinates are otherwise "
                    "plausible.",
                    fix="Check the survey's bin size; override the scalar if needed.",
                    explain="The divisor for the coordinates might be off by a power of ten."))
            elif not ok:
                rep.notes.append(f"Bin/trace spacing is {b:.4g} coordinate units, not one of the common survey sizes. "
                                 "That can be genuine; it is reported for checking.")
    sx2 = sx / f_hdr * eff if cx.type != "float32" else sx
    sy2 = sy / f_hdr * eff if cx.type != "float32" else sy
    if eff != f_hdr:
        bins = _bin_geometry(g, fp, sx2, sy2, zero)
        G.update(bins)
    G["scalar_header"] = hdr_scalar
    G["scalar_effective"] = factor_to_scalar(eff) if cx.type != "float32" else None

    # ---- X/Y swap and CRS -----------------------------------------------------------------
    swap = bool(ov.swap_xy)
    if ov.swap_xy is None and ctype == "projected" and crsmod.looks_swapped(sx2[~zero], sy2[~zero]):
        swap = True
        rep.issues.append(Issue(
            "XY_SWAPPED", "error", "X and Y coordinates are swapped",
            f"The field in bytes {cx.byte} holds values like {np.median(sx2[~zero]):,.0f} (a northing) and bytes {cy.byte} hold "
            f"{np.median(sy2[~zero]):,.0f} (an easting).",
            fix="The fixed file writes easting to X and northing to Y.", auto_fixed=True,
            explain="East-west and north-south positions have been stored in each other's places, which mirrors the survey "
            "across a diagonal and puts it in the wrong place."))
    if swap:
        sx2, sy2 = sy2, sx2
        G.update(_bin_geometry(g, fp, sx2, sy2, zero))
    G["swap_xy"] = swap
    if ctype.startswith("geographic"):
        lon, lat = crsmod.geographic_lonlat(ctype, sx2, sy2)
        if np.nanmax(np.abs(lat)) > 90 >= np.nanmax(np.abs(lon)):
            lon, lat = lat, lon
    pts = ~zero
    cres = crsmod.resolve(sx2[pts], sy2[pts], ctype, tinfo, ov.epsg)
    if cres.status in ("ambiguous", "unknown") and ctype == "projected" and not ov.scalar and cres.candidates and \
            not swap and all(c.inside_area is False for c in cres.candidates[:3]):
        # maybe swapped after all?
        alt = crsmod.resolve(sy2[pts], sx2[pts], ctype, tinfo, ov.epsg)
        if alt.chosen is not None and alt.chosen.inside_area:
            cres = alt
            sx2, sy2 = sy2, sx2
            G["swap_xy"] = True
            G.update(_bin_geometry(g, fp, sx2, sy2, zero))
            rep.issues.append(Issue("XY_SWAPPED", "error", "X and Y coordinates are swapped",
                                    "Only with X and Y exchanged do the coordinates fall inside the stated CRS.",
                                    fix="The fixed file writes them the right way round.", auto_fixed=True,
                                    explain="East and north positions were stored in each other's places."))
    rep.crs = cres.to_dict()
    G["x_min"], G["x_max"] = float(np.min(sx2[pts])) if pts.any() else 0, float(np.max(sx2[pts])) if pts.any() else 0
    G["y_min"], G["y_max"] = float(np.min(sy2[pts])) if pts.any() else 0, float(np.max(sy2[pts])) if pts.any() else 0
    _crs_issues(rep, cres, ctype)

    # ---- plot payload (decimated), outline and lon/lat -------------------------------------
    n = sx2.size
    sel = np.unique(np.linspace(0, n - 1, min(n, 4000)).astype(np.int64))
    rep.plot.update(x=sx2[sel], y=sy2[sel], trace=fp.idx[sel])
    if fp.il is not None:
        rep.plot.update(il=fp.il[sel], xl=fp.xl[sel])
    outline = _outline(g, fp, sx2, sy2, pts, G)
    rep.plot["outline"] = outline
    code = cres.chosen.epsg if cres.chosen else None
    if ctype.startswith("geographic"):
        rep.plot["lonlat"] = np.c_[lon[sel], lat[sel]]
        ox, oy = crsmod.geographic_lonlat(ctype, np.array([p[0] for p in outline]), np.array([p[1] for p in outline])) if outline else ([], [])
        rep.plot["outline_lonlat"] = list(zip(np.asarray(ox).tolist(), np.asarray(oy).tolist()))
    elif code:
        lo, la = crsmod.to_lonlat(code, sx2[sel], sy2[sel])
        rep.plot["lonlat"] = np.c_[lo, la]
        if outline:
            olo, ola = crsmod.to_lonlat(code, np.array([p[0] for p in outline]), np.array([p[1] for p in outline]))
            rep.plot["outline_lonlat"] = list(zip(olo.tolist(), ola.tolist()))
    if rep.plot.get("lonlat") is not None and len(rep.plot["lonlat"]):
        ll = np.asarray(rep.plot["lonlat"])
        clon, clat = float(np.nanmedian(ll[:, 0])), float(np.nanmedian(ll[:, 1]))
        G["centre_lonlat"] = [round(clon, 5), round(clat, 5)]
        try:
            G["centre_on_land"] = bool(crsmod.on_land(clon, clat))
        except Exception:
            pass


def _bin_geometry(g: geo.Geometry, fp: FullPass, sx, sy, zero) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok = ~zero
    if fp.il is not None and fp.xl is not None and g.kind in ("3D post-stack", "3D pre-stack") and ok.sum() >= 3:
        il = fp.il[ok].astype(np.float64)
        xl = fp.xl[ok].astype(np.float64)
        if np.unique(il).size >= 2 and np.unique(xl).size >= 2:
            A = np.column_stack([np.ones_like(il), il, xl])
            cxf, *_ = np.linalg.lstsq(A, sx[ok], rcond=None)
            cyf, *_ = np.linalg.lstsq(A, sy[ok], rcond=None)
            rx = sx[ok] - A @ cxf
            ry = sy[ok] - A @ cyf
            ili = _increment(fp.il)
            xli = _increment(fp.xl)
            d_il = np.array([cxf[1], cyf[1]]) * ili  # moving to the next inline
            d_xl = np.array([cxf[2], cyf[2]]) * xli  # moving to the next crossline (along an inline)
            bin_il = float(np.hypot(*d_il))  # inline spacing (distance between inlines)
            bin_xl = float(np.hypot(*d_xl))
            resid = np.hypot(rx, ry)
            tol = 0.5 * min(bin_il, bin_xl) if min(bin_il, bin_xl) > 0 else np.inf
            out.update(
                inline_spacing=bin_il, crossline_spacing=bin_xl,
                inline_azimuth=float(np.degrees(np.arctan2(d_xl[0], d_xl[1])) % 360),  # direction along an inline
                crossline_azimuth=float(np.degrees(np.arctan2(d_il[0], d_il[1])) % 360),
                grid_rms_residual=float(np.sqrt(np.mean(resid**2))),
                grid_outliers=int((resid > tol).sum()),
                handedness="right" if (d_xl[0] * d_il[1] - d_xl[1] * d_il[0]) < 0 else "left",
                grid_coef_x=cxf.tolist(), grid_coef_y=cyf.tolist(),
                bin_check_size=max(bin_il, bin_xl),
            )
            # survey corners from the exact grid
            il0, il1 = float(fp.il.min()), float(fp.il.max())
            xl0, xl1 = float(fp.xl.min()), float(fp.xl.max())
            corners = []
            for a, b in ((il0, xl0), (il0, xl1), (il1, xl1), (il1, xl0)):
                corners.append({"inline": int(a), "crossline": int(b),
                                "x": float(cxf[0] + cxf[1] * a + cxf[2] * b), "y": float(cyf[0] + cyf[1] * a + cyf[2] * b)})
            out["corners"] = corners
            area = abs((il1 - il0 + ili) / ili * (xl1 - xl0 + xli) / xli) * bin_il * bin_xl if ili and xli else 0
            out["area_km2_grid"] = area / 1e6
            return out
    if ok.sum() >= 2:
        k = fp.key if fp.key is not None else np.arange(sx.size)
        dx = np.diff(sx[ok])
        dy = np.diff(sy[ok])
        dk = np.diff(k[ok]).astype(np.float64)
        d = np.hypot(dx, dy)
        mv = (dk != 0) & (d > 0)
        if mv.any():
            per = d[mv] / np.abs(dk[mv])
            out["trace_spacing"] = float(np.median(per))
            out["line_length"] = float(np.sum(d[(d < 20 * np.median(per) * np.maximum(np.abs(dk), 1))]))
            out["bin_check_size"] = out["trace_spacing"]
            jumps = int((d > 10 * np.median(per) * np.maximum(np.abs(dk), 1)).sum())
            out["position_jumps"] = jumps
    return out


def _outline(g, fp, sx, sy, pts, G) -> list[list[float]]:
    if "corners" in G:
        c = G["corners"]
        return [[p["x"], p["y"]] for p in c] + [[c[0]["x"], c[0]["y"]]]
    if pts.sum() < 2:
        return []
    if g.kind.startswith("2D") or g.kind in ("shot gathers", "pre-stack gathers"):
        sel = np.unique(np.linspace(0, pts.sum() - 1, min(int(pts.sum()), 500)).astype(int))
        return np.c_[sx[pts][sel], sy[pts][sel]].tolist()
    # convex hull for irregular point clouds
    return _hull(np.c_[sx[pts], sy[pts]])


def _hull(p: np.ndarray) -> list[list[float]]:
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p.tolist()
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(tuple(q))
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(tuple(q))
    h = lower[:-1] + upper[:-1]
    return [list(map(float, q)) for q in h + [h[0]]]


def _crs_issues(rep: Report, cres: crsmod.CrsResult, ctype: str) -> None:
    if cres.status == "confirmed":
        return
    if cres.status == "likely":
        rep.issues.append(Issue(
            "CRS_INFERRED", "warning", "Coordinate system worked out, please confirm it",
            f"Most likely EPSG:{cres.chosen.epsg} ({cres.chosen.name}). " + " ".join(cres.notes),
            fix="Confirm the CRS (or pass --crs EPSG:xxxx). The fixed file and load sheet record what was used.",
            explain="The file never says which map projection its coordinates use. Segwise worked out the most likely one "
            "from clues in the text, but a person should confirm it before the data is used."))
    elif cres.status == "ambiguous":
        rep.issues.append(Issue(
            "CRS_AMBIGUOUS", "warning", "Coordinate system needs choosing",
            ("Candidates: " + ", ".join(f"EPSG:{c.epsg} ({c.name})" for c in cres.candidates[:4]) + ". " if cres.candidates else "")
            + " ".join(cres.notes),
            fix="Choose the CRS with --crs EPSG:xxxx (or in the web app). Ask the data supplier if unsure.",
            explain="Several map projections fit the clues equally well, or the clues contradict the coordinates. Loading with "
            "the wrong one shifts the survey by metres to hundreds of kilometres."))
    elif cres.status == "unknown" and ctype == "local":
        rep.issues.append(Issue(
            "LOCAL_COORDINATES", "warning", "Coordinates look like a local grid, not map coordinates",
            "All coordinates are below 1000 and nothing marks them as latitude/longitude.",
            fix="If they are real positions, supply the CRS with --crs; otherwise treat the survey as unpositioned.",
            explain="The X/Y values are too small to be positions on a national or UTM map grid. They may be a local "
            "survey grid or placeholders."))
    elif cres.status == "unknown" and ctype == "projected":
        rep.issues.append(Issue(
            "CRS_UNKNOWN", "warning", "Coordinate system unknown",
            "The file gives no usable CRS clue. " + " ".join(cres.notes),
            fix="Supply it with --crs EPSG:xxxx. Without it the survey can't be placed on a map with confidence.",
            explain="The coordinates are numbers on some map grid, but nothing says which grid."))


def _amplitude_issues(rep: Report, st) -> None:
    if st.n_checked == 0:
        return
    if st.dead_frac >= 0.999:
        rep.issues.append(Issue("ALL_ZERO", "error", "All checked traces are zero",
                                f"All {st.n_checked} sampled traces contain only zeros.",
                                explain="There is no seismic signal in the file, or it was written as blanks."))
    elif st.dead_frac > 0.05:
        rep.issues.append(Issue("DEAD_TRACES", "info", "Many dead (all-zero) traces",
                                f"{st.dead_frac * 100:.1f}% of {st.n_checked} sampled traces are all zero.",
                                explain="Zero traces are normal around the edges of a survey or as padding, but a large share can mean "
                                "missing data."))
    if st.ibm_overflow:
        rep.issues.append(Issue("IBM_OVERFLOW", "warning", "Some IBM samples exceed the IEEE float range",
                                f"{st.ibm_overflow} of the sampled values are valid IBM numbers larger than 3.4e38, which cannot be "
                                "represented as standard IEEE floats.",
                                fix="Kept bit-exact in the default output. If you convert to IEEE (--ieee) they are set to 0 "
                                "and listed in the fix log.",
                                explain="A few amplitude values are astronomically large, almost certainly corrupted words from "
                                "the original tape. Converting the file naively would turn them into 'infinity', which crashes or "
                                "blanks displays in many applications."))
    if st.spikes:
        rep.issues.append(Issue("SPIKES", "info", "Isolated amplitude spikes",
                                f"{st.spikes} sampled values are more than 10,000x the 99th-percentile amplitude (largest "
                                f"{st.spike_max:.3g}).",
                                fix="Consider despiking before amplitude work; display scaling based on the maximum will look blank.",
                                explain="A handful of samples are absurdly loud compared with the rest, usually data errors. "
                                "They make auto-scaled displays look empty."))
    if st.nan_frac > 0 or (st.inf_frac > 0 and not st.ibm_overflow):
        rep.issues.append(Issue("NAN_INF", "error", "Non-numeric sample values (NaN/Inf)",
                                f"{st.nan_frac * 100:.4f}% NaN and {st.inf_frac * 100:.4f}% infinite values in sampled traces.",
                                fix="Replace them (e.g. with zero) before loading; many applications crash or show blank traces.",
                                explain="Some amplitude values are 'not a number', which breaks displays and calculations."))
    if st.clip_frac > 0.002:
        rep.issues.append(Issue("CLIPPED", "warning", "Amplitudes look clipped",
                                f"{st.clip_frac * 100:.2f}% of live samples sit exactly at the maximum absolute value.",
                                explain="The data was saved with its peaks cut off (often after conversion to a small integer "
                                "format), which damages amplitude-based interpretation."))


def _check_claims(rep: Report, t: TextInfo, g: geo.Geometry, fp: FullPass) -> None:
    roles = {"inline": g.inline, "crossline": g.crossline, "cdp": g.cdp, "sp": g.sp, "offset": g.offset}
    if g.coords:
        roles["x"], roles["y"] = g.coords.x, g.coords.y
    hm = {r["byte"]: r for r in rep.header_map}
    for c in t.byte_claims:
        what = c.what
        if what == "scalar":
            continue
        actual = roles.get(what)
        if actual is None:
            if what == "cdp" and g.crossline is not None and c.byte in {g.crossline.byte, *g.crossline.aliases}:
                continue
            continue
        ok = c.byte == actual.byte or c.byte in actual.aliases
        if what in ("x", "y") and not ok and g.coords is not None:
            other = g.coords.y if what == "x" else g.coords.x
            if c.byte == other.byte:
                ok = False
        where = f"line {c.line}"
        if ok:
            rep.geometry.setdefault("evidence", []).append(
                f"The textual header ({where}) says {what} is in byte {c.byte}, confirmed by the data.")
        else:
            inside = hm.get(c.byte)
            desc = f"{inside['pattern']} ({inside['min']:g}..{inside['max']:g})" if inside else "zero/empty"
            rep.issues.append(Issue(
                "TEXT_CLAIM_WRONG", "warning", f"Textual header is wrong about the {what} location",
                f"{where.capitalize()} says the {what} is in byte {c.byte}: \"{c.quote}\". Byte {c.byte} actually contains "
                f"{desc}. The {what} is really in bytes {actual.byte}-{actual.byte + actual.ref.size - 1}.",
                fix="The fixed file moves it to the standard location and the load sheet records the real one.",
                explain="The note at the top of the file tells you where to find this value, and it's wrong. Following it "
                "would load the data with scrambled geometry. This is very common after files are re-processed or re-exported."))
    ranges = {"inline": ("il_min", "il_max"), "crossline": ("xl_min", "xl_max"), "cdp": ("key_min", "key_max"), "sp": ("sp_min", "sp_max")}
    for rc in t.range_claims:
        keys = ranges.get(rc.what)
        G = rep.geometry
        if not keys or keys[0] not in G:
            continue
        lo, hi = G[keys[0]], G[keys[1]]
        if abs(lo - rc.lo) <= 1 and abs(hi - rc.hi) <= 1:
            G.setdefault("evidence", []).append(f"The textual header (line {rc.line}) gives the {rc.what} range "
                                                 f"{rc.lo:g}-{rc.hi:g}, matching the data.")
        else:
            rep.issues.append(Issue(
                "TEXT_RANGE_MISMATCH", "info", f"Textual header {rc.what} range differs from the data",
                f"Line {rc.line} says {rc.what} {rc.lo:g}-{rc.hi:g}; the file contains {lo}-{hi}.",
                explain="The file may be a cut-out of a larger survey, or the text was copied from another file."))
    if t.sample_rate_ms and rep.layout.get("dt_us") and abs(t.sample_rate_ms * 1000 - rep.layout["dt_us"]) > 1:
        rep.issues.append(Issue(
            "TEXT_DT_MISMATCH", "warning", "Textual header sample interval differs",
            f"Text says {t.sample_rate_ms:g} ms; the headers say {rep.layout['dt_us'] / 1000:g} ms.",
            explain="Two parts of the file disagree on the time between samples; check which is right (resampling may have "
            "happened)."))


def _fill_text(rep: Report, t: TextInfo, g) -> None:
    highlights: dict[int, list[str]] = {}
    for c in t.byte_claims:
        if c.line:
            highlights.setdefault(c.line, []).append(f"{c.what} byte {c.byte}")
    for r in t.range_claims:
        highlights.setdefault(r.line, []).append(f"{r.what} range")
    for h in t.hints:
        if h.line:
            highlights.setdefault(h.line, []).append(h.kind.replace("_", " "))
    rep.text = {
        "encoding": rep.layout.get("text_encoding"),
        "lines": t.lines,
        "highlights": {str(k): sorted(set(v)) for k, v in highlights.items()},
        "byte_claims": [c.__dict__ for c in t.byte_claims],
        "range_claims": [r.__dict__ for r in t.range_claims],
        "hints": [{"kind": h.kind, "value": h.value, "line": h.line} for h in t.hints],
        "domain": t.domain, "domain_quote": t.domain_quote,
        "survey_name": t.survey_name, "keywords": t.keywords,
    }
    if t.domain == "depth":
        rep.notes.append(f"The textual header suggests depth-domain data (\"{t.domain_quote}\"). The vertical sample "
                         "interval is then a depth step, not time. Tell your loader so.")


def _findings_geometry(rep: Report, g: geo.Geometry, lay: Layout, claims) -> None:
    G = rep.geometry
    conf = g.confidence
    src = "user" if rep.overrides else "data"
    rep.findings.append(Finding("kind", "Data type", g.kind, g.kind, conf, src, list(g.evidence)))
    def fref(c):
        return f"bytes {c.byte}-{c.byte + c.ref.size - 1} ({c.type})"

    if g.inline is not None:
        rng = f", {G.get('il_min')}-{G.get('il_max')}" if "il_min" in G else ""
        rep.findings.append(Finding("inline", "Inline number", g.inline.ref.to_dict(), fref(g.inline) + rng, conf,
                                    "user" if rep.overrides.get("inline") else "data",
                                    [e for e in G.get("evidence", []) if "inline" in e.lower() or "grid" in e.lower()],
                                    [{"byte": a} for a in g.inline.aliases]))
        rng = f", {G.get('xl_min')}-{G.get('xl_max')}" if "xl_min" in G else ""
        rep.findings.append(Finding("crossline", "Crossline number", g.crossline.ref.to_dict(), fref(g.crossline) + rng, conf,
                                    "user" if rep.overrides.get("crossline") else "data",
                                    [e for e in G.get("evidence", []) if "crossline" in e.lower() or "grid" in e.lower()],
                                    [{"byte": a} for a in g.crossline.aliases]))
    if g.cdp is not None:
        rep.findings.append(Finding("cdp", "CDP / trace number", g.cdp.ref.to_dict(), fref(g.cdp), conf, src, list(g.evidence)))
    if g.ensemble is not None:
        rep.findings.append(Finding("ensemble", "Ensemble (gather) key", g.ensemble.ref.to_dict(), fref(g.ensemble), conf, src,
                                    list(g.evidence)))
    if g.sp is not None:
        rep.findings.append(Finding("sp", "Shotpoint number", g.sp.ref.to_dict(), fref(g.sp), "medium", "data",
                                    ["Varies in step with the CDP number along the line."]))
    if g.coords is not None:
        cc = "high" if g.coords.quality < 0.05 else ("medium" if g.coords.quality < 0.3 else "low")
        ev = [e for e in G.get("evidence", []) if "oordinate" in e]
        if G.get("grid_rms_residual") is not None:
            ev.append(f"Across all traces, positions deviate from a perfect grid by {G['grid_rms_residual']:.3g} units RMS; "
                      f"{G.get('grid_outliers', 0)} traces are more than half a bin off.")
        rep.findings.append(Finding("x", "X coordinate (easting)", g.coords.x.ref.to_dict(), fref(g.coords.x), cc,
                                    "user" if rep.overrides.get("x") else "data", ev, rep.geometry.get("alternatives", [])))
        rep.findings.append(Finding("y", "Y coordinate (northing)", g.coords.y.ref.to_dict(), fref(g.coords.y), cc,
                                    "user" if rep.overrides.get("y") else "data", ev))
        se = G.get("scalar_effective")
        if se is not None:
            ev2 = [G["scalar_evidence"]] if G.get("scalar_evidence") else []
            rep.findings.append(Finding("scalar", "Coordinate scalar", se,
                                        f"{se} (" + ("multiply" if se > 0 else "divide") + f" by {abs(se)})",
                                        "high" if ev2 else "medium",
                                        "user" if rep.overrides.get("scalar") else ("data" if se != G.get("scalar_header") else "binary header"),
                                        ev2 or ["Taken from trace header bytes 71-72."]))
    c = rep.crs or {}
    if c.get("chosen"):
        ch = c["chosen"]
        rep.findings.append(Finding("crs", "Coordinate reference system", ch["epsg"], f"EPSG:{ch['epsg']} - {ch['name']}",
                                    {"confirmed": "high", "likely": "medium", "ambiguous": "low"}.get(c["status"], "low"),
                                    "user" if rep.overrides.get("epsg") else "text header + data", ch["reasons"] + c.get("notes", []),
                                    c.get("candidates", [])[1:6]))
    elif c.get("coord_type") not in (None, "none", "zero"):
        rep.findings.append(Finding("crs", "Coordinate reference system", None, "not determined", "none", "data",
                                    c.get("notes", []), c.get("candidates", [])[:6]))


def _recipe_and_status(rep: Report, lay: Layout, g: geo.Geometry, fmt: int, fp: FullPass) -> None:
    G = rep.geometry
    standard = True
    moves = []
    if g.kind in ("3D post-stack", "3D pre-stack"):
        if g.inline.byte != 189 or g.inline.type != "int32":
            standard = False
            moves.append(f"inline {g.inline.byte} -> 189")
        if g.crossline.byte != 193 or g.crossline.type != "int32":
            standard = False
            moves.append(f"crossline {g.crossline.byte} -> 193")
    if g.kind.startswith("2D post") and g.cdp is not None and (g.cdp.byte != 21 or g.cdp.type != "int32"):
        standard = False
        moves.append(f"CDP {g.cdp.byte} -> 21")
    if g.coords is not None:
        if (g.coords.x.byte, g.coords.y.byte) != (181, 185) or g.coords.x.type != "int32":
            standard = False
            moves.append(f"X/Y {g.coords.x.byte}/{g.coords.y.byte} -> 181/185")
    fixes = [i for i in rep.issues if i.auto_fixed]
    rep.recipe = {
        "standard_layout": standard,
        "moves": moves,
        "auto_fixes": [i.code for i in fixes],
        "output": {
            "endian": "big", "format": fmt if fmt in (1, 2, 3, 5, 8) else 5, "ns": lay.ns, "dt_us": lay.dt_us,
            "inline": {"byte": 189, "type": "int32"} if g.inline is not None else None,
            "crossline": {"byte": 193, "type": "int32"} if g.crossline is not None else None,
            "cdp": {"byte": 21, "type": "int32"} if (g.cdp is not None or g.inline is not None) else None,
            "x": {"byte": 181, "type": "int32"} if g.coords is not None else None,
            "y": {"byte": 185, "type": "int32"} if g.coords is not None else None,
            "scalar": {"byte": 71, "type": "int16"},
        },
    }
    errors = [i for i in rep.issues if i.severity == "error" and not i.auto_fixed]
    needs = [i for i in rep.issues if i.code in ("CRS_AMBIGUOUS", "CRS_UNKNOWN", "CRS_INFERRED", "NO_COORDINATES", "LOCAL_COORDINATES",
                                                "DUPLICATE_TRACES", "BLANK_KEYS")]
    if g.kind == "unknown":
        rep.status = "needs_review"
        rep.status_reason = "The geometry could not be identified automatically; set the key byte locations by hand."
    elif errors:
        rep.status = "needs_review"
        rep.status_reason = errors[0].title
    elif g.confidence == "low":
        rep.status = "needs_review"
        rep.status_reason = "Geometry was identified with low confidence."
    elif needs:
        rep.status = "needs_review"
        rep.status_reason = needs[0].title
    elif fixes or not standard or lay.endian == "<" or lay.is_su:
        rep.status = "fixable"
        rep.status_reason = "Loads with the right settings; a fixed standard copy can be written."
    else:
        rep.status = "ready"
        rep.status_reason = "Standard layout and a confirmed coordinate system. Loads as it is."
    # one-line summary
    parts = [g.kind]
    if "il_min" in G:
        parts.append(f"IL {G['il_min']}-{G['il_max']} x XL {G['xl_min']}-{G['xl_max']}")
    elif "key_min" in G:
        parts.append(f"{'ensembles' if g.ensemble is not None else 'CDP'} {G['key_min']}-{G['key_max']}")
    parts.append(f"{lay.n_traces:,} traces x {lay.ns} samples @ {lay.dt_us / 1000:g} ms")
    if rep.crs.get("chosen"):
        parts.append(f"EPSG:{rep.crs['chosen']['epsg']}" + ("" if rep.crs.get("status") == "confirmed" else " (unconfirmed)"))
    rep.summary = ", ".join(parts)
    key_items = [rep.layout.get("endian"), fmt, lay.ns, lay.dt_us, g.kind]
    for c in (g.inline, g.crossline, g.cdp, g.ensemble):
        key_items.append(c and (c.byte, c.type))
    if g.coords:
        key_items += [(g.coords.x.byte, g.coords.x.type), (g.coords.y.byte, g.coords.y.type), G.get("scalar_effective")]
    rep.layout_key = hashlib.sha1(repr(key_items).encode()).hexdigest()[:10]


def _dedupe(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for i in issues:
        k = (i.code, i.detail)
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


def describe_field(byte: int) -> str:
    f = TRACE_FIELD_BY_BYTE.get(byte)
    return f"{f.name} - {f.label}" if f else "non-standard location"
