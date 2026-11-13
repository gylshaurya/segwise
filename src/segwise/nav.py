"""Merge navigation (positions) from a separate file into SEG-Y trace headers.

Legacy 2D data very often has no coordinates in its trace headers; positions
live in a navigation file keyed by shotpoint, field record (FFID) or CDP.
Supported inputs:
  * UKOOA P1/90 (the industry exchange format: fixed columns, 'H' header records)
  * delimited tables (CSV, TSV, whitespace) with or without a header row,
    including archive formats such as USGS NAMSS navigation (comment lines '!').

Which nav column matches which trace-header field is decided from the data:
every integer nav column is tried against the usual SEG-Y key fields and the
pairing that matches the most traces wins. Gaps are filled by linear
interpolation along the key (reported), never extrapolated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .model import FieldRef

KEY_FIELDS = [(9, "FFID (field record)"), (17, "energy source point"), (197, "shotpoint"), (21, "CDP"), (1, "trace sequence")]


@dataclass
class NavTable:
    lines: np.ndarray  # line name per record ('' when absent)
    keys: dict[str, np.ndarray]  # integer columns usable as keys, by column label
    a: np.ndarray  # x or longitude
    b: np.ndarray  # y or latitude
    geographic: bool
    source_format: str
    columns: dict[str, str] = field(default_factory=dict)


@dataclass
class NavMatch:
    line: str
    nav_key: str
    segy_key: FieldRef
    segy_key_label: str
    matched: int
    interpolated: int
    unmatched: int
    n_traces: int
    to_epsg: int
    nav_epsg: int
    notes: list[str] = field(default_factory=list)
    _k: np.ndarray | None = None
    _x: np.ndarray | None = None
    _y: np.ndarray | None = None

    def lookup(self, keys: np.ndarray):
        keys = np.asarray(keys, np.float64)
        x = np.interp(keys, self._k, self._x, left=np.nan, right=np.nan)
        y = np.interp(keys, self._k, self._y, left=np.nan, right=np.nan)
        x = np.where(np.isnan(x), 0.0, x)
        y = np.where(np.isnan(y), 0.0, y)
        return x, y

    def as_writer_nav(self) -> dict:
        return {"key_ref": self.segy_key.to_dict(), "lookup": self.lookup, "x": self._x, "y": self._y, "units_code": 1}

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "segy_key"} | {
            "segy_key": self.segy_key.to_dict()}


# ------------------------------------------------------------------ parsing
def _p190_angle(s: str, deg_digits: int) -> float | None:
    s = s.strip()
    m = re.match(r"^(\d{%d})(\d{2})(\d{2}(?:\.\d+)?)([NSEW])$" % deg_digits, s)
    if not m:
        return None
    v = int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600
    return -v if m.group(4) in "SW" else v


def parse_p190(text: str) -> NavTable:
    lines, sps, lat, lon, e, n = [], [], [], [], [], []
    for row in text.splitlines():
        if not row or row[0] == "H" or len(row) < 46:
            continue
        if row[0] not in "SVECTRQ":
            continue
        try:
            sp = int(float(row[19:25]))
        except ValueError:
            continue
        la = _p190_angle(row[25:35], 2)
        lo = _p190_angle(row[35:46], 3)
        try:
            ee = float(row[46:55])
            nn = float(row[55:64])
        except ValueError:
            ee = nn = np.nan
        lines.append(row[1:13].strip())
        sps.append(sp)
        lat.append(np.nan if la is None else la)
        lon.append(np.nan if lo is None else lo)
        e.append(ee)
        n.append(nn)
    if not sps:
        raise ValueError("No P1/90 position records found.")
    e, n = np.array(e), np.array(n)
    projected = np.isfinite(e).all() and np.isfinite(n).all() and (np.abs(e) > 0).all()
    return NavTable(np.array(lines), {"shotpoint": np.array(sps, np.int64)},
                    e if projected else np.array(lon), n if projected else np.array(lat),
                    not projected, "UKOOA P1/90",
                    {"shotpoint": "columns 20-25", "x": "easting cols 47-55" if projected else "longitude cols 36-46",
                     "y": "northing cols 56-64" if projected else "latitude cols 26-35"})


def parse_table(text: str, columns: dict[str, str] | None = None) -> NavTable:
    rows = []
    header = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s[0] in "!#;" or s.startswith("//"):
            continue
        toks = [t for t in re.split(r"[,\t;]|\s+", s) if t != ""]
        if header is None and not rows and sum(_is_num(t) for t in toks) < len(toks) / 2:
            header = [t.lower() for t in toks]
            continue
        rows.append(toks)
    if not rows:
        raise ValueError("No data rows found in the navigation file.")
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    rows = [r for r in rows if len(r) == width]
    cols = list(zip(*rows))
    names = header if header and len(header) == width else [f"col{i + 1}" for i in range(width)]
    num, strs = {}, {}
    for name, c in zip(names, cols):
        if all(_is_num(t) for t in c):
            num[name] = np.array([float(t) for t in c])
        else:
            strs[name] = np.array(c)
    columns = {k.lower(): v.lower() for k, v in (columns or {}).items()}

    def named(*opts):
        for o in opts:
            if o in columns and columns[o] in num:
                return columns[o]
        for o in opts:
            if o in num:
                return o
        return None

    lat = named("lat", "latitude")
    lon = named("lon", "long", "longitude")
    x = named("x", "easting", "east", "e")
    y = named("y", "northing", "north", "n")
    if not (lat and lon) and not (x and y):
        floats = {k: v for k, v in num.items() if not np.all(v == np.round(v))}
        latc = [k for k, v in floats.items() if np.all(np.abs(v) <= 90)]
        lonc = [k for k, v in floats.items() if np.all(np.abs(v) <= 180) and k not in latc[:1]]
        if latc and lonc:
            lat, lon = latc[0], lonc[0]
        else:
            big = [k for k, v in floats.items() if np.median(np.abs(v)) > 1000]
            if len(big) >= 2:
                x, y = big[0], big[1]
    if lat and lon:
        a, b, geographic = num[lon], num[lat], True
        used = {lat, lon}
    elif x and y:
        a, b, geographic = num[x], num[y], False
        used = {x, y}
    else:
        raise ValueError("Could not identify coordinate columns (lat/lon or x/y). Name them with --nav-col x=...,y=...")
    keys = {k: v.astype(np.int64) for k, v in num.items()
            if k not in used and np.all(v == np.round(v)) and np.all(np.abs(v) < 2**31)}
    line_col = columns.get("line") if columns.get("line") in strs else (next(iter(strs)) if strs else None)
    lines = strs[line_col] if line_col else np.array([""] * len(a))
    return NavTable(lines, keys, a, b, geographic, "delimited table",
                    {"x": lon or x, "y": lat or y, "line": line_col or ""})


def _is_num(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def read_nav(path: str, columns: dict[str, str] | None = None) -> NavTable:
    with open(path, "r", errors="replace") as f:
        text = f.read()
    if re.search(r"^H0", text, re.M) or re.search(r"^[SV].{18}\s*\d+\s*\d{6}\.\d{2}[NS]", text, re.M):
        try:
            return parse_p190(text)
        except ValueError:
            pass
    return parse_table(text, columns)


# ------------------------------------------------------------------ matching
def name_matches(file_name: str, line: str) -> bool:
    """Does a file name like 'f2860001.sgy' or 'line_12.segy' refer to nav line 'F286-1' / 'L12'?"""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", file_name)
    fd = "".join(re.findall(r"\d", stem))
    groups = re.findall(r"\d+", line)
    if not fd or not groups:
        return False
    head = "".join(groups[:-1])
    if not fd.startswith(head):
        # file names often drop a survey prefix: compare the trailing number only
        m = re.search(r"(\d+)$", stem)
        return bool(m) and int(m.group(1)) == int(groups[-1]) and len(groups) == 1
    rest = fd[len(head):]
    return bool(rest) and int(rest) == int(groups[-1])


def match(report, nav: NavTable, line: str | None = None, key_byte: int | None = None,
          nav_epsg: int | None = None, to_epsg: int | None = None) -> NavMatch:
    from .reader import TraceReader, probe

    lay = probe(report.path)
    r = TraceReader(lay)
    n = lay.n_traces
    segy_keys = {}
    for b, label in KEY_FIELDS:
        if key_byte and b != key_byte:
            continue
        v = r.fields_chunked([(b, "int32")])[0]
        if np.unique(v).size > 1:
            segy_keys[b] = (v, label)
    if not segy_keys:
        raise ValueError("No usable key field (FFID, shotpoint, CDP) varies in the trace headers.")

    names = np.unique(nav.lines)
    pref: list = []
    if line:
        cand_lines = [l for l in names if l.lower() == line.lower()] or [l for l in names if line.lower() in l.lower()]
        if not cand_lines:
            raise ValueError(f"Line '{line}' not found in the navigation file. Lines: {', '.join(names[:20])}")
    else:
        pref = [l for l in names if name_matches(report.file_name, str(l))]
        cand_lines = pref + [l for l in names if l not in pref]

    best = None
    name_pref = set(pref) if not line else set()
    for ln in cand_lines[:400]:
        sel = nav.lines == ln
        for nk, nv in nav.keys.items():
            kv = nv[sel]
            if kv.size < 2:
                continue
            ku = np.unique(kv)
            for b, (sv, label) in segy_keys.items():
                u = np.unique(sv)
                hit = np.isin(u, ku).mean()
                inside = ((u >= ku.min()) & (u <= ku.max())).mean()
                # key numbering restarts on every line, so overlap alone is not enough: the line's extent
                # should also match the file's (similar number of points and same last point)
                size_sim = min(u.size, ku.size) / max(u.size, ku.size)
                end_sim = 1.0 - min(1.0, abs(float(u.max()) - float(ku.max())) / max(float(u.max() - u.min()), 1.0))
                score = hit + 0.25 * inside + 0.5 * size_sim + 0.5 * end_sim + (0.6 if ln in name_pref else 0.0)
                if best is None or score > best[0]:
                    best = (score, ln, nk, b, label, hit, inside)
    if best is None or best[5] < 0.2:
        raise ValueError("The navigation keys do not match the trace headers (no line/column pairing matched even 20% "
                         "of the traces). Specify --nav-line and --nav-key.")
    score, ln, nk, b, label, hit, inside = best
    sel = nav.lines == ln
    k = nav.keys[nk][sel].astype(np.float64)
    a = nav.a[sel]
    bb = nav.b[sel]
    order = np.argsort(k, kind="stable")
    k, a, bb = k[order], a[order], bb[order]
    k, first = np.unique(k, return_index=True)
    a, bb = a[first], bb[first]
    notes = []
    if nav.geographic:
        from pyproj import CRS, Transformer

        src = nav_epsg or 4326
        if not nav_epsg:
            notes.append("The navigation is latitude/longitude with no stated datum; WGS 84 was assumed. For 1980s data "
                         "the datum may be NAD27/WGS 72 (differences of tens to ~200 m). Pass --nav-epsg to set it.")
        if not to_epsg:
            lon0, lat0 = float(np.median(a)), float(np.median(bb))
            zone = int(np.floor((lon0 + 180) / 6)) % 60 + 1
            to_epsg = (32600 if lat0 >= 0 else 32700) + zone
            notes.append(f"Positions were projected to WGS 84 / UTM zone {zone}{'N' if lat0 >= 0 else 'S'} "
                         f"(EPSG:{to_epsg}), the zone containing the line. Pass --to-epsg to choose another CRS.")
        t = Transformer.from_crs(CRS.from_epsg(src), CRS.from_epsg(to_epsg), always_xy=True)
        x, y = t.transform(a, bb)
    else:
        src = nav_epsg or 0
        x, y = a, bb
        if to_epsg and nav_epsg and to_epsg != nav_epsg:
            from pyproj import CRS, Transformer

            x, y = Transformer.from_crs(CRS.from_epsg(nav_epsg), CRS.from_epsg(to_epsg), always_xy=True).transform(a, bb)
        to_epsg = to_epsg or nav_epsg or 0
    sv = segy_keys[b][0]
    u = np.unique(sv)
    exact = np.isin(sv, k)
    within = (sv >= k.min()) & (sv <= k.max())
    m = NavMatch(str(ln), nk, FieldRef(b, "int32"), label, int(exact.sum()), int((within & ~exact).sum()),
                 int((~within).sum()), n, int(to_epsg or 0), int(src or 0), notes, k, np.asarray(x), np.asarray(y))
    m.notes.insert(0, f"Matched trace-header {label} (byte {b}) to navigation column '{nk}' on line '{ln}': "
                      f"{hit * 100:.1f}% of the file's key values appear in the navigation.")
    if m.interpolated:
        m.notes.append(f"{m.interpolated} traces had no exact navigation record and were linearly interpolated "
                       "between neighbouring points.")
    if m.unmatched:
        m.notes.append(f"{m.unmatched} traces lie outside the navigated range and are left with zero coordinates.")
    return m
