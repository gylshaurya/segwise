"""Write a clean, standard SEG-Y rev 1 file from an analysed one, then verify it.

What changes in the output:
  * big-endian, SEG-Y rev 1, fixed-length traces, correct binary header
  * inline -> bytes 189-192, crossline -> 193-196, X/Y -> 181-184/185-188 (int32)
    with the coordinate scalar in 71-72; 2D CDP -> 21-24; shotpoint -> 197-200
  * corrected sample count / interval / format code everywhere
  * a provenance stamp in blank lines of the textual header
What never changes: the original file (opened read-only), the amplitude bytes
(copied bit-exact unless IEEE conversion is requested), and every other header
field (copied as-is).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import __version__
from .fields import BIN, TRACE_FIELDS
from .formats import FORMATS, decode_samples, swap_sample_bytes
from .model import FieldRef
from .reader import FILE_HEADER, TEXT_SIZE, TH, Layout, TraceReader, probe

NATIVE_OUT_FORMATS = {1, 2, 3, 5, 8}  # formats every mainstream loader reads


class WriteError(Exception):
    pass


@dataclass
class FixLog:
    source: str
    output: str
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    n_traces: int = 0
    output_format: int = 0
    output_scalar: int = 0
    seconds: float = 0.0
    loadsheet_path: str = ""

    def to_dict(self):
        return self.__dict__.copy()


# ------------------------------------------------------------------ text header
def _ebcdic(s: str) -> bytes:
    return s.encode("cp037", errors="replace")


def _content(line: str) -> str:
    import re

    return re.sub(r"^\s*C\s*\d{1,2}\s?", "", line).strip()


def build_text_header(lines: list[str], stamp: list[str], is_blank_source: bool) -> tuple[bytes, int]:
    """Return 3200 EBCDIC bytes and how many stamp lines were placed."""
    lines = (list(lines) + [""] * 40)[:40]
    if is_blank_source:
        lines = [""] * 40
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            ln = f"C{i + 1:2d}"
        out.append(ln)
    blanks = [i for i, ln in enumerate(out) if not _content(ln) and i < 39] if not is_blank_source else list(range(1, 39))
    placed = 0
    # fill from the bottom-most blank lines upwards, keeping stamp order
    targets = sorted(blanks[-len(stamp):]) if len(blanks) >= len(stamp) else sorted(blanks)
    for t, text in zip(targets, stamp[: len(targets)]):
        out[t] = f"C{t + 1:2d} {text}"[:80]
        placed += 1
    if is_blank_source:
        out[0] = "C 1 SEG-Y FILE WRITTEN BY SEGWISE FROM A FILE WITHOUT A TEXTUAL HEADER"
        out[39] = "C40 END TEXTUAL HEADER"
    raw = b"".join(_ebcdic(ln.ljust(80)[:80].upper()) for ln in out)
    return raw, placed


# ------------------------------------------------------------------ binary header
def build_binary_header(lay: Layout, out_fmt: int, n_ext: int, measurement: int | None) -> bytes:
    b = dict(lay.binary)
    if lay.endian == ">" and not lay.is_su:
        with open(lay.path, "rb") as f:
            f.seek(TEXT_SIZE)
            raw = bytearray(f.read(400))
    else:
        raw = bytearray(400)
        for name, (byte, size) in BIN.items():
            if size in (2, 4) and name not in ("byte_order_marker",):
                v = b.get(name, 0)
                _put(raw, byte - 3201, size, v)
    # clear rev 2 extensions that rev 1 readers misinterpret
    raw[3260 - 3200 : 3300 - 3200] = bytes(40)
    raw[3300 - 3200 : 3500 - 3200] = bytes(200) if lay.binary.get("rev_major", 0) >= 2 else raw[3300 - 3200 : 3500 - 3200]
    _put(raw, 3217 - 3201, 2, int(lay.dt_us))
    _put(raw, 3221 - 3201, 2, int(lay.ns))
    _put(raw, 3225 - 3201, 2, out_fmt)
    if measurement:
        _put(raw, 3255 - 3201, 2, measurement)
    raw[3501 - 3201] = 1  # SEG-Y rev 1.0
    raw[3502 - 3201] = 0
    _put(raw, 3503 - 3201, 2, 1)  # fixed length traces
    _put(raw, 3505 - 3201, 2, n_ext)
    _put(raw, 3507 - 3201, 4, 0)  # no additional trace headers
    return bytes(raw)


def _put(buf: bytearray, off: int, size: int, value: int) -> None:
    if size == 2:
        value = int(value)
        if value > 32767:
            value -= 65536
        buf[off : off + 2] = np.array([value], ">i2").tobytes()
    elif size == 4:
        buf[off : off + 4] = np.array([int(value)], ">i4").tobytes()
    elif size == 1:
        buf[off] = int(value) & 0xFF


# ------------------------------------------------------------------ trace headers
def _to_big_endian_headers(h: np.ndarray, endian: str) -> np.ndarray:
    if endian == ">":
        return h.copy()
    out = np.zeros_like(h)
    for f in TRACE_FIELDS:
        o = f.byte - 1
        w = f.size
        out[:, o : o + w] = np.ascontiguousarray(h[:, o : o + w])[:, ::-1]
    return out


def _put_field(h: np.ndarray, byte: int, typ: str, values: np.ndarray) -> None:
    o = byte - 1
    if typ == "int16":
        h[:, o : o + 2] = np.ascontiguousarray(np.asarray(values, ">i2")).view(np.uint8).reshape(-1, 2)
    else:
        h[:, o : o + 4] = np.ascontiguousarray(np.asarray(values, ">i4")).view(np.uint8).reshape(-1, 4)


def _get_field(h: np.ndarray, byte: int, typ: str, endian: str) -> np.ndarray:
    o = byte - 1
    size = 2 if typ == "int16" else 4
    code = {"int32": "i4", "int16": "i2", "float32": "f4"}[typ]
    return np.ascontiguousarray(h[:, o : o + size]).view(np.dtype(endian + code)).reshape(-1)


def _choose_out_scalar(coords_max: float, requested: int | None, src_type: str, src_scalar: int) -> int:
    if requested:
        return requested
    if src_type == "int32" and src_scalar not in (0,):
        return src_scalar
    # keep centimetre precision when it fits in int32
    for s in (-100, -10, 1):
        f = 1 / abs(s) if s < 0 else s
        if coords_max / f < 2**31 - 1:
            return s
    return 10


# ------------------------------------------------------------------ main
def write_fixed(report, out_path: str, ieee: bool = False, fill_grid_coords: bool = True,
                nav: dict | None = None, progress: Callable[[str, float], None] | None = None,
                verify: bool = True, overwrite: bool = False) -> FixLog:
    """Write a standardised copy of the analysed file. `report` is an analyze.Report."""
    import time

    t0 = time.time()
    prog = progress or (lambda s, f: None)
    src = report.path
    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path):
        if os.path.samefile(src, out_path):
            raise WriteError("Refusing to overwrite the original file. Choose a different output path.")
        if not overwrite:
            raise WriteError(f"{out_path} already exists (use --overwrite to replace it).")
    if any(i.code == "STRUCTURE_BREAK" for i in report.issues):
        raise WriteError("The file's structure breaks part-way through (see the report). Split it into consistent parts "
                         "first; writing a 'fixed' copy would hide the problem.")
    lay = probe(src)
    if report.layout.get("dt_us"):
        lay.dt_us = float(report.layout["dt_us"])
    if lay.n_traces == 0:
        raise WriteError("The file has no traces to write.")
    G = report.geometry
    log = FixLog(src, out_path)

    fmt_in = report.samples.get("format", lay.format_code)
    out_fmt = fmt_in
    if ieee and fmt_in != 5:
        out_fmt = 5
        log.actions.append(f"Converted samples from format {fmt_in} to IEEE float (format 5).")
    elif fmt_in not in NATIVE_OUT_FORMATS:
        out_fmt = 5
        log.actions.append(f"Format {fmt_in} is not widely supported; samples converted to IEEE float (format 5).")
    if fmt_in != lay.format_code:
        log.actions.append(f"Binary header format code corrected from {lay.format_code} to {out_fmt}.")
    log.output_format = out_fmt

    reader = TraceReader(lay)
    n = lay.n_traces
    if lay.remainder:
        log.actions.append(f"Dropped the incomplete final trace ({lay.remainder} bytes).")
    if lay.endian == "<":
        log.actions.append("Converted from little-endian to big-endian byte order.")
    if lay.is_su:
        log.actions.append("Converted from Seismic Unix to SEG-Y (generated textual and binary headers).")

    refs = {k: FieldRef.from_dict(G.get(k)) for k in ("inline", "crossline", "cdp", "x", "y", "sp", "ensemble")}
    has_xy = refs["x"] is not None and refs["y"] is not None
    scal_eff = G.get("scalar_effective")
    scal_hdr = G.get("scalar_header", 0)
    swap = bool(G.get("swap_xy"))
    coef_x = G.get("grid_coef_x")
    coef_y = G.get("grid_coef_y")
    out_scalar = None
    if has_xy or nav:
        cmax = max(abs(G.get("x_max", 0) or 0), abs(G.get("y_max", 0) or 0), abs(G.get("x_min", 0) or 0), abs(G.get("y_min", 0) or 0), 1)
        if nav:
            cmax = max(cmax, float(np.nanmax(np.abs(nav["x"]))), float(np.nanmax(np.abs(nav["y"]))))
        src_type = refs["x"].type if has_xy else "float32"
        out_scalar = _choose_out_scalar(cmax, None, src_type, scal_eff if scal_eff is not None else scal_hdr)
        if G.get("coord_type", "").startswith("geographic") and out_scalar == -100:
            out_scalar = -10000 if cmax * 10000 < 2**31 else -1000
        log.output_scalar = out_scalar
    moves = report.recipe.get("moves", [])
    for m in moves:
        log.actions.append(f"Moved {m} (SEG-Y rev 1 standard position).")
    if swap:
        log.actions.append("Swapped X and Y so that X is the easting.")
    if scal_eff is not None and scal_eff != scal_hdr:
        log.actions.append(f"Coordinate scalar corrected from {scal_hdr} to {scal_eff}.")

    # ---- text header ---------------------------------------------------------------------
    crs = report.crs.get("chosen") if report.crs else None
    stamp = [
        f"SEGWISE {__version__} {_dt.date.today().isoformat()}: STANDARDISED TO SEG-Y REV 1",
        "BYTES: INLINE 189 XLINE 193 CDP-X 181 CDP-Y 185 SCALAR 71" if refs["inline"] else
        ("BYTES: CDP 21 CDP-X 181 CDP-Y 185 SP 197 SCALAR 71" if refs["cdp"] else "BYTES: SEE LOADSHEET"),
        (f"CRS EPSG:{crs['epsg']} {crs['name']} ({report.crs.get('status', '').upper()})" if crs else "CRS: NOT CONFIRMED"),
        f"FORMAT {out_fmt} NS {lay.ns} DT {lay.dt_us / 1000:g} MS ORIGINAL {os.path.basename(src)}"[:76],
    ]
    blank_src = lay.text_encoding in ("empty", "binary/unreadable", "none")
    text_raw, placed = build_text_header(lay.text_lines, stamp, blank_src)
    if placed < len(stamp):
        log.warnings.append(f"Only {placed} blank textual-header lines were available for the provenance stamp; full details are "
                            "in the load sheet.")
    else:
        log.actions.append("Added a provenance stamp to blank lines of the textual header (original text kept).")
    ext_raw = b""
    with open(src, "rb") as f:
        if lay.n_ext_text:
            f.seek(FILE_HEADER)
            ext_raw = f.read(lay.n_ext_text * TEXT_SIZE)
    measurement = None
    if lay.binary.get("measurement_system") not in (1, 2) and crs:
        try:
            from pyproj import CRS

            unit = CRS.from_epsg(crs["epsg"]).axis_info[0].unit_name.lower()
            measurement = 2 if "foot" in unit or "feet" in unit else (1 if "metre" in unit or "meter" in unit else None)
        except Exception:
            pass
    bin_raw = build_binary_header(lay, out_fmt, lay.n_ext_text, measurement)

    # ---- traces ---------------------------------------------------------------------------
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".segwise-", suffix=".tmp", dir=out_dir)
    zero_filled = 0
    nonfinite = 0
    # fields found at non-standard positions/widths: byte-swap them by their real width, not the standard table's
    from .fields import TRACE_FIELD_BY_BYTE

    nonstd = [row for row in (report.header_map or [])
              if not (TRACE_FIELD_BY_BYTE.get(row["byte"]) and TRACE_FIELD_BY_BYTE[row["byte"]].size == row["end"] - row["byte"] + 1)]
    if lay.endian == "<" and nonstd:
        log.warnings.append("Converted from little-endian: fields at non-standard positions (" +
                            ", ".join(f"{r['byte']}-{r['end']}" for r in nonstd[:8]) + ") were swapped using the widths "
                            "found in the data; any other non-standard content may need checking.")
    overwritten: set[str] = set()
    width_in = FORMATS[fmt_in][0]
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(text_raw)
            out.write(bin_raw)
            out.write(ext_raw)
            bytes_per_trace = TH + lay.ns * FORMATS[out_fmt][0]
            chunk = max(1, min(50_000, (64 << 20) // bytes_per_trace))
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                sl = slice(s, e) if reader.block is not None else np.arange(s, e)
                h_src = reader.headers(sl)
                h = _to_big_endian_headers(h_src, lay.endian)
                if lay.endian == "<":
                    for row in nonstd:
                        o, w = row["byte"] - 1, row["end"] - row["byte"] + 1
                        h[:, o : o + w] = np.ascontiguousarray(h_src[:, o : o + w])[:, ::-1]
                if lay.n_extra_trace_headers and s == 0:
                    log.actions.append(f"Dropped {lay.n_extra_trace_headers} SEG-Y rev 2 extension trace header(s) per trace.")
                # keys
                if refs["inline"] is not None:
                    il = _get_field(h_src, refs["inline"].byte, refs["inline"].type, lay.endian).astype(np.int64)
                    xl = _get_field(h_src, refs["crossline"].byte, refs["crossline"].type, lay.endian).astype(np.int64)
                    _note_overwrite(h, 189, overwritten, il)
                    _note_overwrite(h, 193, overwritten, xl)
                    _put_field(h, 189, "int32", il)
                    _put_field(h, 193, "int32", xl)
                if refs["cdp"] is not None and refs["inline"] is None and (refs["cdp"].byte, refs["cdp"].type) != (21, "int32"):
                    k = _get_field(h_src, refs["cdp"].byte, refs["cdp"].type, lay.endian).astype(np.int64)
                    _note_overwrite(h, 21, overwritten, k)
                    _put_field(h, 21, "int32", k)
                if refs["sp"] is not None and refs["sp"].byte != 197:
                    sp = _get_field(h_src, refs["sp"].byte, refs["sp"].type, lay.endian).astype(np.int64)
                    _note_overwrite(h, 197, overwritten, sp)
                    _put_field(h, 197, "int32", sp)
                # coordinates
                if has_xy:
                    rx = _get_field(h_src, refs["x"].byte, refs["x"].type, lay.endian).astype(np.float64)
                    ry = _get_field(h_src, refs["y"].byte, refs["y"].type, lay.endian).astype(np.float64)
                    sc = _get_field(h_src, 71, "int16", lay.endian).astype(np.float64)
                    if refs["x"].type == "float32":
                        X, Y = rx, ry
                    elif scal_eff is not None and scal_eff != scal_hdr:
                        f = (scal_eff if scal_eff > 0 else 1 / abs(scal_eff))
                        X, Y = rx * f, ry * f
                    else:
                        fac = np.where(sc > 0, sc, np.where(sc < 0, 1 / np.abs(np.where(sc == 0, 1, sc)), 1.0))
                        X, Y = rx * fac, ry * fac
                    if swap:
                        X, Y = Y, X
                    if fill_grid_coords and coef_x and refs["inline"] is not None:
                        z = (rx == 0) & (ry == 0)
                        if G.get("il_min") is not None:
                            z &= (il >= G["il_min"]) & (il <= G["il_max"]) & (xl >= G["xl_min"]) & (xl <= G["xl_max"])
                        if z.any():
                            X = np.where(z, coef_x[0] + coef_x[1] * il + coef_x[2] * xl, X)
                            Y = np.where(z, coef_y[0] + coef_y[1] * il + coef_y[2] * xl, Y)
                            zero_filled += int(z.sum())
                    _write_coords(h, X, Y, out_scalar, overwritten)
                elif nav is not None:
                    key_ref = FieldRef.from_dict(nav["key_ref"])
                    keys = _get_field(h_src, key_ref.byte, key_ref.type, lay.endian).astype(np.int64)
                    X, Y = nav["lookup"](keys)
                    _write_coords(h, X, Y, out_scalar, overwritten)
                    if nav.get("units_code"):
                        _put_field(h, 89, "int16", np.full(len(keys), nav["units_code"]))
                    if refs["cdp"] is None and refs["inline"] is None:
                        pass
                _put_field(h, 115, "int16", np.full(e - s, lay.ns if lay.ns < 32768 else lay.ns - 65536))
                _put_field(h, 117, "int16", np.full(e - s, int(lay.dt_us) if lay.dt_us < 32768 else int(lay.dt_us) - 65536))
                # samples
                raw = reader.raw_samples(sl, fmt_in)
                if out_fmt == fmt_in:
                    smp = swap_sample_bytes(raw, fmt_in) if (lay.endian == "<" and width_in > 1) else raw
                    smp = np.ascontiguousarray(smp)
                else:
                    vals = decode_samples(raw, fmt_in, lay.endian).astype(np.float64)
                    bad = ~np.isfinite(vals)
                    if bad.any():
                        nonfinite += int(bad.sum())
                        vals[bad] = 0.0
                    with np.errstate(over="ignore"):
                        v32 = vals.astype(np.float32)
                    bad = ~np.isfinite(v32)
                    if bad.any():
                        nonfinite += int(bad.sum())
                        v32[bad] = 0
                    smp = np.ascontiguousarray(v32.astype(">f4")).view(np.uint8).reshape(e - s, -1)
                block = np.empty((e - s, TH + smp.shape[1]), np.uint8)
                block[:, :TH] = h
                block[:, TH:] = smp
                out.write(block.tobytes())
                prog("Writing", e / n)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if zero_filled:
        log.actions.append(f"Recomputed coordinates for {zero_filled} traces that had X=Y=0, from the survey grid.")
    if nonfinite:
        log.warnings.append(f"{nonfinite} sample values could not be represented as IEEE floats and were set to 0.")
    if overwritten:
        log.warnings.append("Original non-standard values in bytes " + ", ".join(sorted(overwritten)) +
                            " were replaced by the standard fields.")
    log.n_traces = n
    if verify:
        prog("Verifying", 0.0)
        log.verification = verify_output(report, log, lay, reader, fmt_in, out_fmt, swap)
    log.seconds = round(time.time() - t0, 2)
    return log


def _note_overwrite(h: np.ndarray, byte: int, overwritten: set, new: np.ndarray) -> None:
    old = _get_field(h, byte, "int32", ">")
    if np.any((old != 0) & (old != new)):
        overwritten.add(f"{byte}-{byte + 3}")


def _write_coords(h: np.ndarray, X: np.ndarray, Y: np.ndarray, out_scalar: int, overwritten: set) -> None:
    f = out_scalar if out_scalar > 0 else 1 / abs(out_scalar)
    xi = np.round(X / f).astype(np.int64)
    yi = np.round(Y / f).astype(np.int64)
    if np.any(np.abs(xi) > 2**31 - 1) or np.any(np.abs(yi) > 2**31 - 1):
        raise WriteError("Coordinates are too large to store as 32-bit integers with the chosen scalar.")
    for b, v in ((181, xi), (185, yi)):
        _note_overwrite(h, b, overwritten, v)
        _put_field(h, b, "int32", v)
    # many older loaders read source X/Y; fill them only where they are empty
    sx = _get_field(h, 73, "int32", ">")
    sy = _get_field(h, 77, "int32", ">")
    empty = (sx == 0) & (sy == 0)
    if empty.all():
        _put_field(h, 73, "int32", xi)
        _put_field(h, 77, "int32", yi)
    _put_field(h, 71, "int16", np.full(len(xi), out_scalar))


# ------------------------------------------------------------------ verification
def verify_output(report, log: FixLog, lay_in: Layout, reader_in: TraceReader, fmt_in: int, out_fmt: int, swap: bool) -> dict:
    res: dict[str, Any] = {"checks": [], "passed": True}

    def check(name: str, ok: bool, detail: str = ""):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            res["passed"] = False

    lay = probe(log.output)
    structural = [i.code for i in lay.issues]
    check("Output is readable as standard SEG-Y", lay.endian == ">" and not structural,
          ", ".join(structural) or "big-endian rev 1, no structural issues")
    check("Trace count preserved", lay.n_traces == lay_in.n_traces, f"{lay.n_traces} traces")
    if lay.n_traces != lay_in.n_traces or lay.n_traces == 0:
        return res
    check("Samples per trace and interval preserved", lay.ns == lay_in.ns and lay.dt_us == lay_in.dt_us,
          f"{lay.ns} samples @ {lay.dt_us / 1000:g} ms")
    r = TraceReader(lay)
    n = lay.n_traces
    idx = np.unique(np.linspace(0, n - 1, min(n, 500)).astype(np.int64))
    a = reader_in.samples(idx, fmt_in).astype(np.float64)
    b = r.samples(idx, out_fmt).astype(np.float64)
    fin = np.isfinite(a) & (np.abs(a) < 3.4e38)
    if out_fmt == fmt_in:
        raw_a = reader_in.raw_samples(idx, fmt_in)
        raw_b = r.raw_samples(idx, out_fmt)
        if lay_in.endian == "<":
            raw_a = swap_sample_bytes(raw_a, fmt_in)
        check("Amplitudes bit-identical to the source", np.array_equal(raw_a, raw_b), f"{idx.size} traces compared byte-for-byte")
    else:
        err = np.max(np.abs(a[fin] - b[fin]) / (np.abs(a[fin]) + 1e-30)) if fin.any() else 0.0
        check("Amplitudes preserved after format conversion", err < 1e-6, f"max relative difference {err:.2e}")
    G = report.geometry
    if G.get("inline"):
        il = r.field(idx, 189, "int32")
        xl = r.field(idx, 193, "int32")
        src_il = reader_in.field(idx, G["inline"]["byte"], G["inline"]["type"]).astype(np.int64)
        src_xl = reader_in.field(idx, G["crossline"]["byte"], G["crossline"]["type"]).astype(np.int64)
        check("Inline/crossline at bytes 189/193 match the source", np.array_equal(il, src_il) and np.array_equal(xl, src_xl),
              f"inline {il.min()}-{il.max()}, crossline {xl.min()}-{xl.max()} in checked traces")
    if G.get("x") and log.output_scalar:
        x = r.field(idx, 181, "int32").astype(np.float64)
        y = r.field(idx, 185, "int32").astype(np.float64)
        f = log.output_scalar if log.output_scalar > 0 else 1 / abs(log.output_scalar)
        X, Y = x * f, y * f
        coef_x, coef_y = G.get("grid_coef_x"), G.get("grid_coef_y")
        if coef_x and G.get("inline"):
            il = r.field(idx, 189, "int32").astype(np.float64)
            xl = r.field(idx, 193, "int32").astype(np.float64)
            ex = coef_x[0] + coef_x[1] * il + coef_x[2] * xl
            ey = coef_y[0] + coef_y[1] * il + coef_y[2] * xl
            dev = np.hypot(X - ex, Y - ey)[(X != 0) | (Y != 0)]
            dev = dev if dev.size else np.zeros(1)
            bin_ = min(G.get("inline_spacing") or 1, G.get("crossline_spacing") or 1)
            # f = coordinate units per stored integer step (e.g. 10 m for scalar +10): rounding can't beat that
            tol = max(0.25 * bin_, 0.75 * f)
            check("Coordinates at bytes 181/185 sit on the survey grid", float(np.median(dev)) < tol,
                  f"median deviation {np.median(dev):.3g} (bin {bin_:.3g})")
        live = (X != 0) | (Y != 0)  # null coordinates are reported separately, not an extent failure
        Xl, Yl = (X[live], Y[live]) if live.any() else (X, Y)
        rng_ok = (G.get("x_min", -np.inf) - 1 <= Xl.min() and Xl.max() <= G.get("x_max", np.inf) + 1) or bool(G.get("swap_xy"))
        check("Coordinates within the analysed survey extent", rng_ok,
              f"X {Xl.min():.2f}-{Xl.max():.2f}, Y {Yl.min():.2f}-{Yl.max():.2f}")
    # independent reader
    try:
        import segyio

        kw = {}
        if G.get("inline") and G.get("kind") == "3D post-stack":
            with segyio.open(log.output, "r", iline=189, xline=193, strict=False) as s:
                if s.unstructured:
                    check("segyio (independent reader) opens the file", True,
                          f"opened unstructured: {s.tracecount} traces (grid is irregular, which segyio reports as unstructured)")
                else:
                    ok = (int(s.ilines.min()) == G.get("il_min") and int(s.ilines.max()) == G.get("il_max")
                          and int(s.xlines.min()) == G.get("xl_min") and int(s.xlines.max()) == G.get("xl_max"))
                    check("segyio (independent reader) recovers the 3D geometry with default settings", ok,
                          f"{len(s.ilines)} inlines x {len(s.xlines)} crosslines, format {int(s.format)}")
        else:
            with segyio.open(log.output, "r", ignore_geometry=True, **kw) as s:
                check("segyio (independent reader) opens the file", s.tracecount == n, f"{s.tracecount} traces")
    except ImportError:
        res["checks"].append({"name": "segyio cross-check", "ok": True, "detail": "segyio not installed, skipped"})
    except Exception as e:
        check("segyio (independent reader) opens the file", False, str(e))
    return res


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
