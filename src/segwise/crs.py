"""Coordinate reference system (CRS) detective work.

Map coordinates in SEG-Y headers are just numbers; they mean nothing without
the CRS (datum + projection + zone). The CRS is almost never stored in a
machine-readable way -- at best it is typed into the textual header.

This module:
  * classifies the coordinates (projected metres/feet, geographic degrees,
    arc-seconds, DMS, all-zero),
  * turns textual hints ("ED50 UTM ZONE 31N") into EPSG candidates using
    pyproj's offline EPSG database,
  * checks each candidate against the data: do the coordinates fall inside
    the CRS's area of use, and inside any region the text mentions?
  * when the zone is missing, tries the zones that overlap the mentioned
    region and keeps those that put the survey in that region.
Datums cannot be told apart from coordinates alone (they differ by ~10-300 m),
so a datum that is not stated is always reported as an open question.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

import numpy as np

from .textheader import REGIONS, TextInfo

try:  # pyproj is a hard dependency, but keep import errors readable
    from pyproj import CRS, Transformer
    from pyproj.database import query_crs_info
except Exception:  # pragma: no cover
    CRS = None


@dataclass
class CrsCandidate:
    epsg: int
    name: str
    reasons: list[str] = field(default_factory=list)
    inside_area: bool | None = None
    in_region: list[str] = field(default_factory=list)
    out_region: list[str] = field(default_factory=list)
    centre_lonlat: tuple[float, float] | None = None
    score: float = 0.0
    from_text: bool = False

    def to_dict(self):
        return {
            "epsg": self.epsg, "name": self.name, "reasons": self.reasons, "inside_area_of_use": self.inside_area,
            "matches_regions": self.in_region, "conflicts_regions": self.out_region,
            "centre_lonlat": [round(v, 5) for v in self.centre_lonlat] if self.centre_lonlat else None,
            "score": round(self.score, 2), "from_text": self.from_text,
        }


@dataclass
class CrsResult:
    coord_type: str  # "projected" | "geographic-degrees" | "geographic-arcsec" | "geographic-dms" | "zero" | "none"
    status: str  # "confirmed" | "likely" | "ambiguous" | "unknown" | "not-applicable"
    chosen: CrsCandidate | None
    candidates: list[CrsCandidate]
    notes: list[str] = field(default_factory=list)
    datum_stated: bool = False
    units: str | None = None  # "metre" | "feet" | "degree"

    def to_dict(self):
        return {
            "coord_type": self.coord_type, "status": self.status,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "candidates": [c.to_dict() for c in self.candidates[:8]], "notes": self.notes,
            "datum_stated": self.datum_stated, "units": self.units,
        }


# ------------------------------------------------------------------ EPSG name index
@functools.lru_cache(maxsize=1)
def _index():
    infos = query_crs_info(auth_name="EPSG", pj_types=["PROJECTED_CRS", "GEOGRAPHIC_2D_CRS"])
    by_name = {}
    by_code = {}
    for i in infos:
        if i.deprecated:
            continue
        code = int(i.code)
        by_name[_norm(i.name)] = code
        by_code[code] = i
    return by_name, by_code


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def lookup(name: str) -> int | None:
    return _index()[0].get(_norm(name))


def crs_name(code: int) -> str:
    info = _index()[1].get(code)
    if info is not None:
        return info.name
    try:
        return CRS.from_epsg(code).name
    except Exception:
        return f"EPSG:{code}"


def area_of_use(code: int):
    info = _index()[1].get(code)
    if info is not None and info.area_of_use is not None:
        a = info.area_of_use
        return (a.west, a.south, a.east, a.north)
    try:
        a = CRS.from_epsg(code).area_of_use
        return (a.west, a.south, a.east, a.north) if a else None
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def _to_lonlat(code: int):
    return Transformer.from_crs(CRS.from_epsg(code), CRS.from_epsg(4326), always_xy=True)


def to_lonlat(code: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = _to_lonlat(code)
    lon, lat = t.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def is_geographic(code: int) -> bool:
    try:
        return CRS.from_epsg(code).is_geographic
    except Exception:
        return False


# ------------------------------------------------------------------ coordinate type
def classify(x: np.ndarray, y: np.ndarray, units_code: int, geographic_hint: bool = False) -> str:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or (np.all(x == 0) and np.all(y == 0)):
        return "zero"
    ax, ay = np.nanmax(np.abs(x)), np.nanmax(np.abs(y))
    if units_code == 2 and ax <= 180 * 3600 and ay <= 90 * 3600:
        return "geographic-arcsec"
    if units_code == 3 and ax <= 180 and ay <= 90:
        return "geographic-degrees"
    if units_code == 4:
        return "geographic-dms"
    if geographic_hint and ax <= 180 and ay <= 90 and (np.ptp(x) < 5 and np.ptp(y) < 5):
        # small numbers alone are not evidence of latitude/longitude (test files and local grids look the same)
        return "geographic-degrees"
    if ax < 1000 and ay < 1000:
        return "local"
    return "projected"


def dms_to_deg(v: np.ndarray) -> np.ndarray:
    """SEG-Y DMS coordinates are DDDMMSS.ss encoded as a number."""
    v = np.asarray(v, float)
    s = np.sign(v)
    a = np.abs(v)
    d = np.floor(a / 10000)
    m = np.floor((a - d * 10000) / 100)
    sec = a - d * 10000 - m * 100
    return s * (d + m / 60 + sec / 3600)


def geographic_lonlat(coord_type: str, x: np.ndarray, y: np.ndarray):
    if coord_type == "geographic-arcsec":
        return np.asarray(x) / 3600.0, np.asarray(y) / 3600.0
    if coord_type == "geographic-dms":
        return dms_to_deg(x), dms_to_deg(y)
    return np.asarray(x, float), np.asarray(y, float)


def looks_swapped(x: np.ndarray, y: np.ndarray) -> bool:
    """UTM-style eastings are 166 000-834 000; northings up to 10 000 000."""
    mx, my = float(np.median(x)), float(np.median(y))
    return 1_000_000 < mx < 10_000_000 and 100_000 < my < 900_000


# ------------------------------------------------------------------ candidates
_UTM_FAMILIES = {
    "WGS 84": "WGS 84 / UTM zone {z}{h}",
    "ED50": "ED50 / UTM zone {z}{h}",
    "ETRS89": "ETRS89 / UTM zone {z}{h}",
    "NAD27": "NAD27 / UTM zone {z}{h}",
    "NAD83": "NAD83 / UTM zone {z}{h}",
    "WGS 72": "WGS 72 / UTM zone {z}{h}",
    "SAD69": "SAD69 / UTM zone {z}{h}",
    "SIRGAS 2000": "SIRGAS 2000 / UTM zone {z}{h}",
    "PSAD56": "PSAD56 / UTM zone {z}{h}",
    "Indian 1975": "Indian 1975 / UTM zone {z}{h}",
    "Indian 1960": "Indian 1960 / UTM zone {z}{h}",
    "Minna": "Minna / UTM zone {z}{h}",
    "Arc 1960": "Arc 1960 / UTM zone {z}{h}",
    "Kertau 1968": "Kertau 1968 / UTM zone {z}{h}",
    "Timbalai 1948": "Timbalai 1948 / UTM zone {z}{h}",
    "Camacupa 1948": "Camacupa 1948 / UTM zone {z}{h}",
    "Pulkovo 1942": "Pulkovo 1942 / Gauss-Kruger zone {z}",
    "GDA94": "GDA94 / MGA zone {z}",
    "GDA2020": "GDA2020 / MGA zone {z}",
    "AGD66": "AGD66 / AMG zone {z}",
    "AGD84": "AGD84 / AMG zone {z}",
    "NZGD2000": "NZGD2000 / UTM zone {z}{h}",
    "ED79": "ED79 / UTM zone {z}{h}",
    "ED87": "ED87 / UTM zone {z}{h}",
}
_GEOG = {"WGS 84": 4326, "ED50": 4230, "NAD27": 4267, "NAD83": 4269, "ETRS89": 4258, "GDA94": 4283, "GDA2020": 7844,
         "SAD69": 4618, "SIRGAS 2000": 4674, "OSGB36": 4277, "Pulkovo 1942": 4284, "WGS 72": 4322, "Amersfoort": 4289}
_FIXED = {"OSGB36": 27700, "Amersfoort": 28992}
# Datums commonly used for legacy data in a region, offered when the text names no datum.
_REGIONAL_DATUMS = [
    ((-30, 34, 45, 82), ["ED50", "WGS 84", "ETRS89"]),
    ((-170, 15, -50, 85), ["NAD27", "NAD83", "WGS 84"]),
    ((-180, 50, -129, 72), ["NAD27", "NAD83", "WGS 84"]),
    ((110, -45, 155, -9), ["GDA94", "AGD84", "WGS 84"]),
]


def _utm_zone(lon: float) -> int:
    return int(np.floor((lon + 180) / 6)) % 60 + 1


def _zone_name(datum: str, zone: int, hemi: str) -> int | None:
    fam = _UTM_FAMILIES.get(datum)
    if fam is None:
        return None
    return lookup(fam.format(z=zone, h=hemi))


def _region_boxes(text: TextInfo | None) -> list[tuple[str, tuple]]:
    if text is None:
        return []
    out, seen = [], set()
    for h in text.hints_of("region"):
        if h.value not in seen:
            seen.add(h.value)
            out.append((h.value, REGIONS[h.value]))
    return out


def _in_box(lon: float, lat: float, box, margin: float = 1.0) -> bool:
    w, s, e, n = box
    if not (s - margin) <= lat <= (n + margin):
        return False
    lon = (lon + 180.0) % 360.0 - 180.0
    if w <= e:
        return (w - margin) <= lon <= (e + margin)
    return lon >= (w - margin) or lon <= (e + margin)  # box crosses the 180 degree meridian


def evaluate(code: int, x: np.ndarray, y: np.ndarray, regions: list[tuple[str, tuple]]) -> CrsCandidate:
    cand = CrsCandidate(code, crs_name(code))
    try:
        lon, lat = to_lonlat(code, x, y)
    except Exception as e:  # pragma: no cover - pyproj failure
        cand.reasons.append(f"could not transform: {e}")
        return cand
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        cand.inside_area = False
        cand.reasons.append("coordinates cannot be converted to latitude/longitude in this CRS")
        return cand
    clon, clat = float(np.median(lon[ok])), float(np.median(lat[ok]))
    cand.centre_lonlat = (clon, clat)
    aou = area_of_use(code)
    if aou is not None:
        w, s, e, n = aou
        inside = (w - 0.5 <= clon <= e + 0.5) and (s - 0.5 <= clat <= n + 0.5) if w <= e else (clon >= w - 0.5 or clon <= e + 0.5) and (s - 0.5 <= clat <= n + 0.5)
        cand.inside_area = bool(inside)
    for name, box in regions:
        (cand.in_region if _in_box(clon, clat, box) else cand.out_region).append(name)
    return cand


def resolve(x: np.ndarray, y: np.ndarray, coord_type: str, text: TextInfo | None, user_epsg: int | None = None) -> CrsResult:
    regions = _region_boxes(text)
    hints = text.hints if text else []
    datums = list(dict.fromkeys(h.value for h in hints if h.kind == "datum"))
    zones = list(dict.fromkeys(h.value for h in hints if h.kind == "utm_zone"))
    epsgs = list(dict.fromkeys(h.value for h in hints if h.kind == "epsg"))
    cms = [h.value for h in hints if h.kind == "cm"]
    projs = {h.value for h in hints if h.kind == "projection"}
    units = next((h.value for h in hints if h.kind == "units"), None)
    res = CrsResult(coord_type, "unknown", None, [], datum_stated=bool(datums or epsgs), units=units)

    if coord_type == "local":
        res.status = "unknown"
        res.notes.append("Coordinates are small numbers (under 1000) with no units code or text saying they are "
                         "latitude/longitude: they look like a local or synthetic grid. No real-world CRS can be assigned.")
        return res
    if coord_type in ("zero", "none"):
        res.status = "not-applicable"
        res.notes.append("There are no usable coordinates in the trace headers, so no CRS can be checked.")
        return res

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size > 400:
        sel = np.linspace(0, x.size - 1, 400).astype(int)
        x, y = x[sel], y[sel]

    cands: dict[int, CrsCandidate] = {}

    def add(code: int | None, reason: str, from_text: bool, weight: float):
        if not code:
            return
        if code not in cands:
            if coord_type.startswith("geographic") != is_geographic(code):
                return
            gx, gy = (x, y) if coord_type == "projected" else geographic_lonlat(coord_type, x, y)
            c = evaluate(code, gx, gy, regions)
            c.from_text = from_text
            cands[code] = c
        cands[code].reasons.append(reason)
        cands[code].score += weight

    if user_epsg:
        add(user_epsg, "set by the user", False, 100)
    for code in epsgs:
        add(code, f"EPSG:{code} is written in the textual header", True, 10)

    if coord_type.startswith("geographic"):
        for d in datums or ["WGS 84"]:
            add(_GEOG.get(d), f"geographic coordinates on datum {d}" + (" (stated)" if datums else " (assumed)"),
                bool(datums), 5 if datums else 1)
    else:
        for d in datums:
            if d in _FIXED:
                add(_FIXED[d], f"{d} named in the textual header", True, 6)
        zone_list = list(zones)
        if not zone_list and cms:
            zone_list = [(_utm_zone(cm), None) for cm in cms]
        if zone_list:
            for (z, h) in zone_list:
                for hemi in ([h] if h else ["N", "S"]):
                    for d in (datums or [None]):
                        if d is None:
                            continue
                        add(_zone_name(d, z, hemi), f"{d} + zone {z}{hemi} stated in the textual header", True, 8)
                    if not datums:
                        for d in _guess_datums(regions):
                            add(_zone_name(d, z, hemi), f"zone {z}{hemi} stated; datum {d} assumed (not stated)", False, 2)
        if not zone_list and ("UTM" in projs or datums or regions):
            # zone not given: try zones overlapping the regions the text mentions
            for name, box in regions:
                for lon in np.arange(box[0], box[2] + 0.01, 3.0):
                    z = _utm_zone(lon)
                    hemi = "N" if (box[1] + box[3]) / 2 >= 0 else "S"
                    for d in (datums or _guess_datums(regions)):
                        add(_zone_name(d, z, hemi), f"zone {z}{hemi} overlaps '{name}' mentioned in the text"
                            + ("" if datums else f"; datum {d} assumed"), bool(datums), 1)

    # score by agreement with the data
    for c in cands.values():
        if c.inside_area is False:
            c.score -= 6
            c.reasons.append("coordinates fall OUTSIDE this CRS's area of use")
        elif c.inside_area:
            c.score += 1
        c.score += 3 * len(c.in_region) - 2 * len(c.out_region)
        if c.in_region:
            c.reasons.append(f"places the survey inside {', '.join(c.in_region)} as the text says")
        if c.out_region and not c.in_region:
            c.reasons.append(f"places the survey outside {', '.join(c.out_region)} named in the text")

    ranked = sorted(cands.values(), key=lambda c: -c.score)
    res.candidates = ranked
    if not ranked:
        if coord_type == "projected":
            res.notes.append("The textual header gives no usable CRS information. The coordinates look like a projected "
                             "grid (metres or feet), but the projection, zone and datum must be supplied by the user.")
        res.status = "unknown"
        return res
    top = ranked[0]
    good_top = top.inside_area is not False and not (top.out_region and not top.in_region)
    if user_epsg and top.epsg == user_epsg:
        res.chosen, res.status = top, "confirmed" if good_top else "ambiguous"
        if not good_top:
            res.notes.append("The CRS you chose does not place the survey where expected. Check it.")
        return res
    rivals = [c for c in ranked[1:] if c.score >= top.score - 0.5]
    if top.from_text and good_top and res.datum_stated and not rivals and top.score >= 8:
        res.chosen, res.status = top, "confirmed"
    elif good_top and top.score > 0:
        res.chosen, res.status = top, ("likely" if not rivals else "ambiguous")
    else:
        res.chosen, res.status = None, "ambiguous" if ranked else "unknown"
    if not res.datum_stated and coord_type == "projected":
        res.notes.append("The datum is not stated. Candidates on different datums differ by roughly 10-300 m on the "
                         "ground; confirm the datum with the data supplier before using positions for drilling or permits.")
    if rivals and res.status != "confirmed":
        res.notes.append("Several CRSs fit equally well: " + ", ".join(f"EPSG:{c.epsg}" for c in [top] + rivals[:4]) + ".")
    return res


def _guess_datums(regions) -> list[str]:
    if not regions:
        return ["WGS 84"]
    out: list[str] = []
    for _, box in regions:
        clon, clat = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for (w, s, e, n), ds in _REGIONAL_DATUMS:
            if w <= clon <= e and s <= clat <= n:
                out += ds
    return list(dict.fromkeys(out or ["WGS 84"]))


# ------------------------------------------------------------------ land/sea
@functools.lru_cache(maxsize=1)
def _land():
    from importlib.resources import files

    d = np.load(files("segwise").joinpath("data/land50m.npz"))
    xy, lens = d["xy"], d["lens"]
    rings = np.split(xy, np.cumsum(lens)[:-1])
    from matplotlib.path import Path

    boxes = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in rings])
    return rings, boxes, [Path(r) for r in rings]


def land_rings():
    return _land()[0]


def on_land(lon: float, lat: float) -> bool:
    rings, boxes, paths = _land()
    hit = np.where((boxes[:, 0] <= lon) & (boxes[:, 2] >= lon) & (boxes[:, 1] <= lat) & (boxes[:, 3] >= lat))[0]
    return any(paths[i].contains_point((lon, lat)) for i in hit)
