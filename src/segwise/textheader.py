"""Deterministic reading of the SEG-Y textual header ("EBCDIC header").

The textual header is 40 lines of free text written by whoever produced the
file. It *often* states where the inline/crossline numbers and coordinates are
stored, which coordinate reference system (CRS) is used and where the survey
is, but there is no fixed syntax and it is frequently wrong or stale.

Everything extracted here is treated as a *claim* to be tested against the
data, never as truth. Each claim keeps its line number and quote so the report
can show exactly where it came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Claim:
    what: str  # "inline" | "crossline" | "x" | "y" | "cdp" | "sp" | "offset" | "scalar"
    byte: int
    line: int  # 1-based line number in the textual header
    quote: str


@dataclass
class RangeClaim:
    what: str
    lo: float
    hi: float
    line: int
    quote: str


@dataclass
class Hint:
    kind: str  # "epsg" | "datum" | "utm_zone" | "projection" | "units" | "region" | "spheroid" | "cm"
    value: object
    line: int
    quote: str


@dataclass
class TextInfo:
    lines: list[str]
    byte_claims: list[Claim] = field(default_factory=list)
    range_claims: list[RangeClaim] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    domain: str | None = None  # "time" | "depth"
    domain_quote: str | None = None
    sample_rate_ms: float | None = None
    survey_name: str | None = None
    keywords: list[str] = field(default_factory=list)

    def claims_for(self, what: str) -> list[Claim]:
        return [c for c in self.byte_claims if c.what == what]

    def hints_of(self, kind: str) -> list[Hint]:
        return [h for h in self.hints if h.kind == kind]


# ------------------------------------------------------------------- vocabulary
_LABELS: list[tuple[str, str]] = [
    # order matters: longer / more specific first
    ("x", r"\b(?:CDP|CMP|BIN|ENSEMBLE)[\s_\-]*X\b|\bX[\s_\-]*(?:COORD(?:INATE)?S?|LOC(?:ATION)?)\b|\bEASTINGS?\b|\bCDPX\b|\bX[\s_\-]*CDP\b|\bSRC[\s_\-]*X\b|\bSOURCE[\s_\-]*X\b|\bX[\s_\-]*(?=\s*[:=]?\s*(?:BYTES?\s*)?\d)"),
    ("y", r"\b(?:CDP|CMP|BIN|ENSEMBLE)[\s_\-]*Y\b|\bY[\s_\-]*(?:COORD(?:INATE)?S?|LOC(?:ATION)?)\b|\bNORTHINGS?\b|\bCDPY\b|\bY[\s_\-]*CDP\b|\bSRC[\s_\-]*Y\b|\bSOURCE[\s_\-]*Y\b|\bY[\s_\-]*(?=\s*[:=]?\s*(?:BYTES?\s*)?\d)"),
    ("crossline", r"\b(?:CROSS[\s_\-]?LINES?|X[\s_\-]?LINES?|XLINES?|XL|XLN|TRACE[\s_\-]?(?:NO|NUMBER|NUM)|CROSSLINE)\b"),
    ("inline", r"\b(?:IN[\s_\-]?LINES?|ILINES?|IL|INL|ILN|INLINE)\b"),
    ("cdp", r"\b(?:CDP|CMP|ENSEMBLE)(?:[\s_\-]*(?:NO\.?|NUMBER|NUM|#))?\b"),
    ("sp", r"\b(?:SHOT[\s_\-]?POINTS?|SP|SHOTPOINT|SHOT[\s_\-]?(?:NO|NUMBER))\b"),
    ("offset", r"\bOFFSETS?\b"),
    ("scalar", r"\b(?:CO-?ORD(?:INATE)?[\s_\-]*SCALAR|SCALAR|SCALE[\s_\-]*FACTOR)\b"),
]
_LABEL_RE = [(w, re.compile(p, re.I)) for w, p in _LABELS]
_BYTE_CTX = re.compile(r"\bBYTES?\b|\bBYT\b|\bLOC(?:ATION)?S?\b|\bPOS(?:ITION)?S?\b|\bHEADER\b|\bHDR\b|\bTRC\s*HDR\b|\bWORD\b", re.I)
_NUM_RANGE = re.compile(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:-|–|\.\.|TO|:)\s*(-?\d+(?:\.\d+)?)(?![\d.])", re.I)
_NUM = re.compile(r"(?<![\d.\-])(\d{1,3})(?![\d.])")

DATUMS = {
    # pattern -> canonical name used in the pyproj/EPSG database
    r"\bWGS[\s\-_]*84\b|\bWGS[\s\-_]*1984\b": "WGS 84",
    r"\bWGS[\s\-_]*72\b": "WGS 72",
    r"\bED[\s\-_]*50\b|\bEUROPEAN[\s_]*DATUM[\s_]*(?:19)?50\b|\bED[\s\-_]*1950\b": "ED50",
    r"\bED[\s\-_]*79\b": "ED79",
    r"\bED[\s\-_]*87\b": "ED87",
    r"\bETRS[\s\-_]*89\b": "ETRS89",
    r"\bNAD[\s\-_]*27\b|\bNORTH[\s_]*AMERICAN[\s_]*DATUM[\s_]*(?:19)?27\b": "NAD27",
    r"\bNAD[\s\-_]*83\b|\bNORTH[\s_]*AMERICAN[\s_]*DATUM[\s_]*(?:19)?83\b": "NAD83",
    r"\bGDA[\s\-_]*94\b": "GDA94",
    r"\bGDA[\s\-_]*2020\b": "GDA2020",
    r"\bAGD[\s\-_]*66\b": "AGD66",
    r"\bAGD[\s\-_]*84\b": "AGD84",
    r"\bSAD[\s\-_]*69\b": "SAD69",
    r"\bSIRGAS[\s\-_]*2000\b": "SIRGAS 2000",
    r"\bPSAD[\s\-_]*56\b": "PSAD56",
    r"\bPULKOVO\b": "Pulkovo 1942",
    r"\bINDIAN[\s_]*1975\b": "Indian 1975",
    r"\bINDIAN[\s_]*1960\b": "Indian 1960",
    r"\bMINNA\b": "Minna",
    r"\bARC[\s_]*1960\b": "Arc 1960",
    r"\bKERTAU\b": "Kertau 1968",
    r"\bTIMBALAI\b": "Timbalai 1948",
    r"\bOSGB[\s\-_]*(?:19)?36\b|\bBRITISH[\s_]*NATIONAL[\s_]*GRID\b": "OSGB36",
    r"\bAMERSFOORT\b|\bRD[\s_]*NEW\b": "Amersfoort",
    r"\bNZGD[\s\-_]*2000\b": "NZGD2000",
    r"\bNZGD[\s\-_]*49\b": "NZGD49",
    r"\bCAMACUPA\b": "Camacupa 1948",
    r"\bHUNGARIAN\b": "HD72",
    r"\bNGO[\s\-_]*48\b": "NGO 1948",
    r"\bDEALUL[\s_]*PISCULUI\b": "Dealul Piscului 1970",
}
SPHEROIDS = {
    r"\bINTERNATIONAL[\s_]*(?:1924|1909|SPHEROID|ELLIPSOID)\b|\bHAYFORD\b": "International 1924",
    r"\bCLARKE[\s_]*1866\b": "Clarke 1866",
    r"\bCLARKE[\s_]*1880\b": "Clarke 1880",
    r"\bGRS[\s\-_]*80\b": "GRS 1980",
    r"\bBESSEL\b": "Bessel 1841",
    r"\bKRASSOVSKY\b|\bKRASOVSKY\b": "Krassowsky 1940",
    r"\bEVEREST\b": "Everest",
    r"\bAIRY\b": "Airy 1830",
}
PROJECTIONS = {
    r"\bU\.?T\.?M\.?\b|\bUNIVERSAL[\s_]*TRANSVERSE[\s_]*MERCATOR\b": "UTM",
    r"\bTRANSVERSE[\s_]*MERCATOR\b|\bGAUSS[\s\-_]*KR[UÜ]GER\b": "Transverse Mercator",
    r"\bLAMBERT\b|\bLCC\b": "Lambert Conformal Conic",
    r"\bSTATE[\s_]*PLANE\b|\bSPCS\b": "State Plane",
    r"\bMGA\b": "MGA",
    r"\bGEOGRAPHIC\b|\bLAT(?:ITUDE)?[\s/_]*LON(?:G(?:ITUDE)?)?\b": "Geographic",
}

# Rough lon/lat boxes used to sanity-check where a survey lands.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (lon_min, lat_min, lon_max, lat_max)
    "NORTH SEA": (-4.5, 50.5, 12.5, 62.0),
    "NORWEGIAN SEA": (-2.0, 62.0, 17.0, 72.0),
    "BARENTS SEA": (15.0, 68.0, 60.0, 82.0),
    "IRISH SEA": (-7.0, 51.5, -2.8, 55.0),
    "CELTIC SEA": (-11.0, 48.0, -4.0, 52.5),
    "WEST OF SHETLAND": (-9.0, 58.5, -1.5, 62.5),
    "ATLANTIC MARGIN": (-20.0, 50.0, 5.0, 66.0),
    "NETHERLANDS": (2.5, 50.7, 7.3, 55.9),
    "DUTCH": (2.5, 50.7, 7.3, 55.9),
    "DENMARK": (3.0, 54.5, 15.5, 58.0),
    "DANISH": (3.0, 54.5, 15.5, 58.0),
    "NORWAY": (1.0, 56.0, 35.0, 81.0),
    "UKCS": (-10.0, 49.0, 4.0, 62.5),
    "UNITED KINGDOM": (-10.0, 49.0, 4.0, 62.5),
    "GERMANY": (3.0, 47.0, 15.5, 55.5),
    "GULF OF MEXICO": (-98.0, 18.0, -80.0, 31.0),
    "GOM": (-98.0, 18.0, -80.0, 31.0),
    "TEXAS": (-107.0, 25.5, -93.0, 37.0),
    "LOUISIANA": (-94.5, 28.0, -88.5, 33.5),
    "OKLAHOMA": (-103.5, 33.5, -94.0, 37.5),
    "WYOMING": (-111.5, 40.5, -104.0, 45.5),
    "COLORADO": (-109.5, 36.5, -101.5, 41.5),
    "CALIFORNIA": (-125.5, 32.0, -114.0, 42.5),
    "ALASKA": (-180.0, 50.0, -129.0, 72.0),
    "BERING SEA": (162.0, 51.0, -157.0, 66.5),  # west > east: crosses the 180 degree meridian
    "BEAUFORT": (-160.0, 69.0, -120.0, 76.0),
    "CHUKCHI": (-180.0, 66.0, -155.0, 75.0),
    "NORTH SLOPE": (-166.0, 68.0, -140.0, 71.5),
    "ALBERTA": (-120.5, 48.5, -109.5, 60.5),
    "NOVA SCOTIA": (-67.0, 41.0, -57.0, 47.5),
    "NEWFOUNDLAND": (-60.0, 40.0, -44.0, 56.0),
    "BRAZIL": (-74.0, -34.0, -28.0, 5.5),
    "CAMPOS": (-42.5, -24.5, -38.5, -20.5),
    "SANTOS": (-48.5, -28.5, -41.5, -22.5),
    "TRINIDAD": (-62.5, 9.5, -58.5, 11.8),
    "VENEZUELA": (-73.5, 0.5, -59.5, 13.0),
    "ARGENTINA": (-74.0, -56.0, -53.0, -21.5),
    "NIGERIA": (2.5, 3.0, 15.0, 14.0),
    "NIGER DELTA": (4.0, 3.0, 9.0, 6.5),
    "ANGOLA": (8.0, -18.5, 24.5, -4.0),
    "GABON": (7.5, -4.5, 14.8, 2.5),
    "EGYPT": (24.5, 21.5, 37.0, 32.5),
    "LIBYA": (9.0, 19.5, 25.5, 34.0),
    "ALGERIA": (-9.0, 18.5, 12.5, 37.5),
    "RED SEA": (32.0, 12.0, 44.0, 30.0),
    "PERSIAN GULF": (47.5, 23.5, 57.0, 30.5),
    "ARABIAN GULF": (47.5, 23.5, 57.0, 30.5),
    "OMAN": (51.5, 16.5, 60.0, 26.5),
    "CASPIAN": (46.5, 36.5, 55.5, 47.5),
    "BLACK SEA": (27.0, 40.5, 42.0, 47.0),
    "MEDITERRANEAN": (-6.0, 30.0, 36.5, 46.0),
    "ADRIATIC": (12.0, 39.5, 20.0, 46.0),
    "INDIA": (67.0, 6.0, 98.0, 36.0),
    "KRISHNA": (79.5, 14.5, 84.0, 17.5),
    "BAY OF BENGAL": (79.0, 5.0, 95.0, 23.0),
    "MALAY": (99.0, 0.0, 120.0, 8.5),
    "SOUTH CHINA SEA": (103.0, 0.0, 122.0, 24.0),
    "BOHAI": (117.5, 37.0, 122.5, 41.0),
    "INDONESIA": (94.0, -11.5, 141.5, 6.5),
    "NW SHELF": (110.0, -23.0, 126.0, -12.0),
    "NORTH WEST SHELF": (110.0, -23.0, 126.0, -12.0),
    "AUSTRALIA": (112.0, -44.5, 154.5, -9.0),
    "BROWSE": (119.0, -16.5, 126.0, -12.0),
    "CARNARVON": (110.0, -26.0, 118.0, -18.0),
    "BASS STRAIT": (143.0, -41.5, 150.0, -37.5),
    "GIPPSLAND": (145.5, -39.5, 150.0, -37.5),
    "TARANAKI": (170.5, -41.5, 175.5, -37.5),
    "NEW ZEALAND": (165.0, -48.0, 179.9, -33.5),
    "CANTERBURY": (170.0, -46.0, 175.5, -42.0),
    "GREENLAND": (-75.0, 59.0, -10.0, 84.0),
}

_REGION_RE = {name: re.compile(r"\b" + re.escape(name).replace(r"\ ", r"[\s_\-]+") + r"\b", re.I) for name in REGIONS}


def _clean(line: str) -> str:
    # drop the conventional "C12 " card prefix
    return re.sub(r"^\s*C\s*\d{1,2}\s?", "", line)


def _is_byte_number(n: int) -> bool:
    return 1 <= n <= 240


def parse(lines: list[str]) -> TextInfo:
    info = TextInfo(lines=lines)
    seen_claims: set[tuple[str, int]] = set()
    in_byte_section = False

    for ln_no, raw in enumerate(lines, start=1):
        line = _clean(raw)
        up = line.upper()
        if not up.strip():
            continue
        if re.search(r"BYTE\s*(?:LOC|POS)|HEADER\s*(?:POS|LOC|MAP|BYTE)|TRACE\s*HEADER|HEADER\s*WORDS|HDR\s*LOC", up):
            in_byte_section = True
        byte_ctx = bool(_BYTE_CTX.search(up)) or in_byte_section

        # --- byte location claims -----------------------------------------------------------
        labels = []
        taken: list[tuple[int, int]] = []
        for what, rx in _LABEL_RE:
            for m in rx.finditer(up):
                if any(m.start() < e and m.end() > s for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                labels.append((m.start(), m.end(), what))
        labels.sort()
        for i, (s, e, what) in enumerate(labels):
            nxt = labels[i + 1][0] if i + 1 < len(labels) else len(up)
            seg = up[e:nxt]
            seg = seg[:40]
            rng = _NUM_RANGE.search(seg)
            if rng:
                a, b = float(rng.group(1)), float(rng.group(2))
                if byte_ctx and a.is_integer() and b.is_integer() and _is_byte_number(int(a)) and 0 < b - a <= 4:
                    key = (what, int(a))
                    if key not in seen_claims:
                        seen_claims.add(key)
                        info.byte_claims.append(Claim(what, int(a), ln_no, raw.strip()))
                    continue
                if b - a > 4 or not byte_ctx or a > 240:
                    if what in ("inline", "crossline", "cdp", "sp"):
                        info.range_claims.append(RangeClaim(what, min(a, b), max(a, b), ln_no, raw.strip()))
                    continue
            if byte_ctx:
                m = _NUM.search(seg)
                if m and _is_byte_number(int(m.group(1))):
                    n = int(m.group(1))
                    # a single number right after a label in a byte context is a byte position;
                    # bare numbers outside any byte context are too ambiguous to use.
                    key = (what, n)
                    if key not in seen_claims:
                        seen_claims.add(key)
                        info.byte_claims.append(Claim(what, n, ln_no, raw.strip()))

        # --- CRS hints ----------------------------------------------------------------------
        for m in re.finditer(r"\bEPSG\s*(?:CODE)?\s*[:#=]?\s*(\d{4,6})\b", up):
            info.hints.append(Hint("epsg", int(m.group(1)), ln_no, raw.strip()))
        for pat, name in DATUMS.items():
            if re.search(pat, up):
                info.hints.append(Hint("datum", name, ln_no, raw.strip()))
        for pat, name in SPHEROIDS.items():
            if re.search(pat, up):
                info.hints.append(Hint("spheroid", name, ln_no, raw.strip()))
        for pat, name in PROJECTIONS.items():
            if re.search(pat, up):
                info.hints.append(Hint("projection", name, ln_no, raw.strip()))
        for m in re.finditer(r"\bZONE\s*[:#=]?\s*(\d{1,2})\s*([NS]|NORTH|SOUTH)?\b|\bUTM\s*[\-_]?\s*(\d{1,2})\s*([NS]|NORTH|SOUTH)?\b|\b(\d{1,2})\s*([NS])\b(?=.*\bUTM\b)", up):
            z = m.group(1) or m.group(3) or m.group(5)
            h = m.group(2) or m.group(4) or m.group(6)
            zi = int(z)
            if 1 <= zi <= 60:
                hemi = None if not h else ("N" if h.startswith("N") else "S")
                info.hints.append(Hint("utm_zone", (zi, hemi), ln_no, raw.strip()))
        for m in re.finditer(r"\b(?:CM|C\.M\.|CENTRAL\s*MERIDIAN|CENT\.?\s*MER\.?|LONG(?:ITUDE)?\s*OF\s*ORIGIN)\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)\s*(?:DEG(?:REES?)?)?\s*([EW])?", up):
            v = float(m.group(1))
            if m.group(2) == "W":
                v = -abs(v)
            info.hints.append(Hint("cm", v, ln_no, raw.strip()))
        if re.search(r"\b(?:INT(?:ERNATIONAL)?\.?\s*)?FEET\b|\bFT\b|\bUS\s*SURVEY\s*FEET\b", up):
            info.hints.append(Hint("units", "feet", ln_no, raw.strip()))
        if re.search(r"\bMETRES?\b|\bMETERS?\b", up):
            info.hints.append(Hint("units", "metre", ln_no, raw.strip()))
        address_like = bool(re.search(r"@|\(\d{3}\)|\b\d{3}[\s\-.]\d{3}[\s\-.]?\d{4}\b|\b[A-Z]{2,}\s+\d{5}(?:-\d{4})?\b|"
                                      r"\b(?:ROAD|RD\.|STREET|ST\.|SUITE|AVENUE|AVE\.?|P\.?\s?O\.?\s*BOX|TEL|PHONE|FAX|"
                                      r"CONTACT|HOUSTON|DENVER|CALGARY|ABERDEEN|STAVANGER|PERTH|LONDON|PROCESSED\s+BY)\b", up))
        for name, rx in _REGION_RE.items():
            if rx.search(up) and not address_like:
                info.hints.append(Hint("region", name, ln_no, raw.strip()))

        # --- domain / sample rate / misc -----------------------------------------------------
        if info.domain is None:
            if re.search(r"\bDEPTH\s*(?:DOMAIN|VOLUME|DATA|CUBE|SECTION|CONVERTED)\b|\bPSDM\b|\bPRE[\s\-]*STACK\s*DEPTH|\bDEPTH\s*MIGRAT|\bZ\s*UNITS?\s*[:=]?\s*(?:M|METRES?|METERS?|FT|FEET)\b|\bSAMPLE\s*(?:INTERVAL|RATE)\s*[:=]?\s*\d+(?:\.\d+)?\s*(?:M|METRES?|METERS?|FT|FEET)\b", up):
                info.domain, info.domain_quote = "depth", raw.strip()
            elif re.search(r"\bTWT\b|\bTWO[\s\-]*WAY\s*TIME\b|\bTIME\s*(?:DOMAIN|MIGRAT)|\bPSTM\b|\bPRE[\s\-]*STACK\s*TIME", up):
                info.domain, info.domain_quote = "time", raw.strip()
        m = re.search(r"\bSAMPLE\s*(?:INTERVAL|RATE)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:MS|MSEC|MILLISEC)", up)
        if m and info.sample_rate_ms is None:
            info.sample_rate_ms = float(m.group(1))
        if info.survey_name is None:
            m = re.search(r"\b(?:SURVEY|PROJECT|PROSPECT|AREA)\s*(?:NAME)?\s*[:=]\s*(.{3,50})$", line, re.I)
            if m:
                info.survey_name = m.group(1).strip()
        for kw in ("MIGRATION", "KIRCHHOFF", "PSTM", "PSDM", "STACK", "RTM", "AGC", "DECON", "DMO", "NMO", "FINAL", "RAW",
                   "FILTER", "MULTIPLE", "DEMULTIPLE", "4D", "PRESTACK", "PRE-STACK", "GATHERS", "ANGLE", "OFFSET", "VELOCITY"):
            if re.search(r"\b" + re.escape(kw) + r"\b", up) and kw not in info.keywords:
                info.keywords.append(kw)
    return info
