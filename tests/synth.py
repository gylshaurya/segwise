"""Generate SEG-Y files with known ground truth, in every awkward layout real deliveries contain.

Each `Case` describes a survey (grid, rotation, bin size, origin) and how the file
was (mis)written: where inline/crossline/coordinates live, which scalar, byte
order, sample format, textual header contents, truncation, swapped X/Y, etc.
`write(case, path)` produces the file; the case object is the answer key.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segwise.formats import float_to_ibm


@dataclass
class Case:
    name: str
    kind: str = "3d"  # "3d" | "2d" | "shots"
    n_il: int = 30
    n_xl: int = 40
    il0: int = 1000
    xl0: int = 2000
    il_inc: int = 1
    xl_inc: int = 1
    ns: int = 200
    dt_us: int = 4000
    origin: tuple[float, float] = (620000.0, 6074000.0)  # easting, northing (metres)
    bin_il: float = 25.0  # distance between inlines
    bin_xl: float = 12.5  # distance between crosslines
    azimuth: float = 30.0  # degrees, direction along an inline (increasing crossline)
    il_byte: tuple[int, str] = (189, "int32")
    xl_byte: tuple[int, str] = (193, "int32")
    x_byte: tuple[int, str] = (181, "int32")
    y_byte: tuple[int, str] = (185, "int32")
    cdp_byte: int = 21
    scalar: int = -100  # written scalar
    true_scalar: int | None = None  # scalar the raw ints actually need (None = same as `scalar`)
    endian: str = ">"
    fmt: int = 1
    declared_fmt: int | None = None
    text: list[str] = field(default_factory=list)
    text_ascii: bool = False
    truncate: int = 0  # bytes cut from the end
    swap_xy: bool = False
    sort_by_xl: bool = False
    missing_frac: float = 0.0
    zero_coord_frac: float = 0.0
    bin_ns: int | None = None  # value written in binary header samples/trace (None = ns)
    trace_ns: int | None = None
    epsg: int | None = 23031  # true CRS of the coordinates (for checks)
    seed: int = 7
    extra: dict = field(default_factory=dict)

    @property
    def effective_scalar(self) -> int:
        return self.true_scalar if self.true_scalar is not None else self.scalar


def _traces(n: int, ns: int, rng) -> np.ndarray:
    t = np.arange(ns)
    refl = np.zeros((n, ns))
    for k in range(12):
        pos = rng.integers(10, ns - 10)
        refl[:, pos] += rng.normal(0, 1)
    # band-limited: convolve with a Ricker wavelet
    w = np.arange(-15, 16)
    f = 0.08
    ricker = (1 - 2 * (np.pi * f * w) ** 2) * np.exp(-((np.pi * f * w) ** 2))
    out = np.array([np.convolve(r, ricker, mode="same") for r in refl])
    out += rng.normal(0, 0.02, out.shape)
    out *= 1000 * (1 + 0.3 * np.sin(t / 40.0))
    return out.astype(np.float32)


def geometry(c: Case):
    """Return per-trace arrays (il, xl, x, y) in file order, plus the kept mask."""
    rng = np.random.default_rng(c.seed)
    if c.kind == "2d":
        n = c.n_xl
        cdp = np.arange(c.xl0, c.xl0 + n * c.xl_inc, c.xl_inc)
        s = np.arange(n) * c.bin_xl
        az = np.radians(c.azimuth)
        # gently curving line
        x = c.origin[0] + s * np.sin(az) + 200 * np.sin(s / 3000.0)
        y = c.origin[1] + s * np.cos(az)
        return cdp, None, x, y
    il = c.il0 + np.arange(c.n_il) * c.il_inc
    xl = c.xl0 + np.arange(c.n_xl) * c.xl_inc
    if c.sort_by_xl:
        XL, IL = np.meshgrid(xl, il, indexing="ij")
    else:
        IL, XL = np.meshgrid(il, xl, indexing="ij")
    IL, XL = IL.ravel(), XL.ravel()
    keep = np.ones(IL.size, bool)
    if c.missing_frac:
        # cut a corner off, like an irregular survey outline
        i_n = (IL - c.il0) / max(c.n_il - 1, 1)
        x_n = (XL - c.xl0) / max(c.n_xl - 1, 1)
        keep = (i_n + x_n) > c.missing_frac * 2
    IL, XL = IL[keep], XL[keep]
    a = np.radians(c.azimuth)
    ux = np.array([np.sin(a), np.cos(a)])  # along inline (xl increasing)
    uy = np.array([np.sin(a + np.pi / 2), np.cos(a + np.pi / 2)])  # across (il increasing)
    di = (IL - c.il0) / c.il_inc * c.bin_il
    dx = (XL - c.xl0) / c.xl_inc * c.bin_xl
    x = c.origin[0] + dx * ux[0] + di * uy[0]
    y = c.origin[1] + dx * ux[1] + di * uy[1]
    return IL, XL, x, y


def write(c: Case, path: str) -> dict:
    rng = np.random.default_rng(c.seed)
    il, xl, x, y = geometry(c)
    n = il.size
    e = c.endian
    data = _traces(n, c.ns, rng)
    th = np.zeros((n, 240), np.uint8)

    def put(byte: int, typ: str, vals):
        o = byte - 1
        dt = np.dtype(e + {"int32": "i4", "int16": "i2", "float32": "f4"}[typ])
        size = 2 if typ == "int16" else 4
        th[:, o : o + size] = np.ascontiguousarray(np.asarray(vals).astype(dt)).view(np.uint8).reshape(n, size)

    put(1, "int32", np.arange(1, n + 1))
    put(5, "int32", np.arange(1, n + 1))
    put(115, "int16", np.full(n, c.trace_ns if c.trace_ns is not None else c.ns))
    put(117, "int16", np.full(n, c.dt_us))
    put(29, "int16", np.ones(n))
    eff = c.effective_scalar
    f = eff if eff > 0 else 1 / abs(eff)
    X, Y = (y, x) if c.swap_xy else (x, y)
    zero = np.zeros(n, bool)
    if c.zero_coord_frac:
        zero[rng.choice(n, int(n * c.zero_coord_frac), replace=False)] = True
    if c.kind == "2d":
        put(c.cdp_byte, "int32", il)
        put(197, "int32", (il // 2))
    elif c.kind == "shots":
        ffid = 1 + np.arange(n) // 48
        chan = 1 + np.arange(n) % 48
        put(9, "int32", ffid)
        put(13, "int32", chan)
        put(37, "int32", chan * 25)
        put(73, "int32", np.round(x / f))
        put(77, "int32", np.round(y / f))
    else:
        put(c.il_byte[0], c.il_byte[1], il)
        put(c.xl_byte[0], c.xl_byte[1], xl)
        put(c.cdp_byte, "int32", np.arange(1, n + 1)) if c.cdp_byte not in (c.il_byte[0], c.xl_byte[0]) else None
    if c.kind in ("3d", "2d"):
        if c.x_byte[1] == "float32":
            put(c.x_byte[0], "float32", np.where(zero, 0, X))
            put(c.y_byte[0], "float32", np.where(zero, 0, Y))
        else:
            put(c.x_byte[0], "int32", np.where(zero, 0, np.round(X / f)))
            put(c.y_byte[0], "int32", np.where(zero, 0, np.round(Y / f)))
    put(71, "int16", np.full(n, c.scalar))
    put(89, "int16", np.ones(n))

    if c.fmt == 1:
        words = float_to_ibm(data)
        raw = words.astype(e + "u4").view(np.uint8).reshape(n, -1)
    elif c.fmt == 5:
        raw = data.astype(e + "f4").view(np.uint8).reshape(n, -1)
    elif c.fmt == 2:
        raw = np.round(data).astype(e + "i4").view(np.uint8).reshape(n, -1)
    elif c.fmt == 3:
        raw = np.clip(np.round(data / 4), -32768, 32767).astype(e + "i2").view(np.uint8).reshape(n, -1)
    elif c.fmt == 8:
        raw = np.clip(np.round(data / 40), -128, 127).astype("i1").view(np.uint8).reshape(n, -1)
    else:
        raise ValueError(c.fmt)

    lines = (c.text + [""] * 40)[:40]
    txt = "".join(f"C{i + 1:2d} {ln}".ljust(80)[:80] for i, ln in enumerate(lines))
    text = txt.encode("ascii") if c.text_ascii else txt.encode("cp037")
    b = bytearray(400)

    def bput(byte, size, v):
        b[byte - 3201 : byte - 3201 + size] = np.array([v], e + ("i2" if size == 2 else "i4")).tobytes()

    bput(3217, 2, c.dt_us)
    bput(3221, 2, c.bin_ns if c.bin_ns is not None else c.ns)
    bput(3225, 2, c.declared_fmt or c.fmt)
    bput(3229, 2, 4)
    b[3501 - 3201] = 1
    bput(3503, 2, 1)
    with open(path, "wb") as fh:
        fh.write(text)
        fh.write(bytes(b))
        blob = np.concatenate([th, raw], axis=1).tobytes()
        if c.truncate:
            blob = blob[: -c.truncate]
        fh.write(blob)
    return {"il": il, "xl": xl, "x": x, "y": y, "data": data}


TEXT_GOOD = ["ACME GEOPHYSICAL - NORTH SEA 3D REPROCESSING", "DATUM ED50  UTM ZONE 31N  EPSG:23031",
             "TRACE HEADER BYTE LOCATIONS:", "INLINE 189-192  XLINE 193-196  CDP X 181-184  CDP Y 185-188",
             "SAMPLE INTERVAL 4 MS", "", "END TEXTUAL HEADER"]


def cases() -> list[Case]:
    """The torture suite."""
    T = TEXT_GOOD
    return [
        Case("standard", text=T),
        Case("legacy_9_21_73_77", il_byte=(9, "int32"), xl_byte=(21, "int32"), x_byte=(73, "int32"), y_byte=(77, "int32"),
             cdp_byte=25, text=["NORTH SEA SURVEY", "ED50 UTM 31N"]),
        Case("nonstandard_221_225", il_byte=(221, "int32"), xl_byte=(225, "int32"), x_byte=(201, "int32"),
             y_byte=(205, "int32"), text=["DUTCH NORTH SEA"]),
        Case("int16_keys", il_byte=(181, "int16"), xl_byte=(185, "int16"), x_byte=(73, "int32"), y_byte=(77, "int32"),
             text=["NORTH SEA"]),
        Case("little_endian", endian="<", text=T),
        Case("ieee", fmt=5, text=T),
        Case("ieee_declared_ibm", fmt=5, declared_fmt=1, text=T),
        Case("ibm_declared_ieee", fmt=1, declared_fmt=5, text=T),
        Case("int16_samples", fmt=3, text=T),
        Case("wrong_scalar", scalar=1, true_scalar=-100, text=T),
        Case("swapped_xy", swap_xy=True, text=T),
        Case("crossline_sorted", sort_by_xl=True, text=T),
        Case("irregular_outline", missing_frac=0.3, text=T),
        Case("zero_coords", zero_coord_frac=0.02, text=T),
        Case("truncated", truncate=500, text=T),
        Case("bin_ns_wrong", bin_ns=0, text=T),
        Case("trace_ns_stale", trace_ns=1500, text=T),
        Case("lying_text", il_byte=(9, "int32"), xl_byte=(21, "int32"), cdp_byte=25, text=T),
        Case("ascii_text", text_ascii=True, text=T),
        Case("no_text", text=[]),
        Case("float_coords", x_byte=(201, "float32"), y_byte=(205, "float32"), text=["NORTH SEA ED50 UTM 31N"]),
        Case("il_step_2", il_inc=2, xl_inc=4, text=T),
        Case("big_survey_geometry", n_il=120, n_xl=150, ns=50, bin_il=25, bin_xl=25, azimuth=-12, text=T),
        Case("wide_survey", n_il=40, n_xl=700, ns=40, text=T),  # rows longer than the header-sampling blocks
        Case("line_2d", kind="2d", n_xl=900, bin_xl=12.5, text=["2D LINE NORTH SEA ED50 UTM 31N"]),
        Case("shot_gathers", kind="shots", n_il=1, n_xl=480, text=["RAW SHOTS"]),
        Case("feet_gom", origin=(1_900_000.0, 10_200_000.0), bin_il=82.5, bin_xl=82.5, scalar=-10, epsg=None,
             text=["GULF OF MEXICO NAD27 STATE PLANE FEET"]),
    ]
