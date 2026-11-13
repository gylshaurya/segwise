"""Find out from the numbers themselves what each trace-header field contains.

The approach in plain terms:

1. *Scan*: read every 2- and 4-byte position of the 240-byte trace header, as
   integers (and as floats where that makes sense), for a sample of traces taken
   in contiguous blocks. Describe each one: constant? counts up? repeats?
2. *Keys*: a post-stack 3D volume is a grid. Exactly one pair of fields will
   behave like (inline, crossline): together they are unique for every trace,
   one stays constant for a run of traces while the other counts through its
   range and then resets.
3. *Coordinates*: on a real grid, map X and Y are an exact affine function of
   (inline, crossline):  X = x0 + a*IL + b*XL.  Fitting that to every candidate
   field and finding two fields that fit to rounding precision is extremely
   strong proof -- it is very hard for wrong fields to fit a plane exactly.
4. If there is no grid it tries a 2D line (one key + smoothly moving
   coordinates) and then pre-stack gathers (repeating ensemble keys).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fields import COORD_PAIR_PREFERENCE, INLINE_PREFERENCE, XLINE_PREFERENCE, TRACE_FIELD_BY_BYTE, trace_field_at
from .model import FieldRef


# --------------------------------------------------------------------------- scan
@dataclass
class Cand:
    byte: int
    type: str
    values: np.ndarray  # int64 or float64 over the sampled traces
    nunique: int
    vmin: float
    vmax: float
    change_frac: float  # fraction of consecutive (within-block) pairs that differ
    step: float  # most common non-zero consecutive difference
    step_frac: float  # fraction of non-zero differences equal to `step`
    aliases: list[int] = field(default_factory=list)

    @property
    def ref(self) -> FieldRef:
        return FieldRef(self.byte, self.type)

    @property
    def constant(self) -> bool:
        return self.nunique <= 1

    @property
    def std_name(self) -> str | None:
        f = TRACE_FIELD_BY_BYTE.get(self.byte)
        if f and f.size == (2 if self.type == "int16" else 4):
            return f.name
        return None

    @property
    def aligned(self) -> bool:
        """Standard SEG-Y 4-byte fields start at 1-based bytes 1, 5, 9, ... (or are standard fields)."""
        if self.type == "int16":
            return True
        return (self.byte - 1) % 4 == 0 or self.std_name is not None


def _describe(byte: int, typ: str, v: np.ndarray, blocks: np.ndarray) -> Cand:
    same_block = blocks[1:] == blocks[:-1]
    d = np.diff(v)[same_block] if v.size > 1 else np.zeros(0)
    nz = d[d != 0]
    if nz.size:
        vals, cnt = np.unique(nz, return_counts=True)
        step = float(vals[cnt.argmax()])
        step_frac = float(cnt.max() / nz.size)
    else:
        step, step_frac = 0.0, 0.0
    u = np.unique(v)
    return Cand(byte, typ, v, int(u.size), float(u[0]) if u.size else 0.0, float(u[-1]) if u.size else 0.0,
                float((d != 0).mean()) if d.size else 0.0, step, step_frac)


def scan(headers: np.ndarray, endian: str, blocks: np.ndarray) -> list[Cand]:
    """Describe every plausible field in the sampled trace headers (uint8 array (m, 240))."""
    h = np.ascontiguousarray(headers)
    out: list[Cand] = []
    i16 = np.dtype(endian + "i2")
    i32 = np.dtype(endian + "i4")
    f32 = np.dtype(endian + "f4")
    for o in range(0, 240, 2):
        v16 = np.ascontiguousarray(h[:, o : o + 2]).view(i16).reshape(-1).astype(np.int64)
        out.append(_describe(o + 1, "int16", v16, blocks))
        if o + 4 <= 240:
            v32 = np.ascontiguousarray(h[:, o : o + 4]).view(i32).reshape(-1).astype(np.int64)
            c = _describe(o + 1, "int32", v32, blocks)
            # an int16 sitting in the high half of a zero-padded int32 shows up as value*65536
            if not (c.nunique > 1 and np.all(v32 % 65536 == 0)):
                out.append(c)
            if (o % 4) == 0:
                with np.errstate(all="ignore"):
                    vf = np.ascontiguousarray(h[:, o : o + 4]).view(f32).reshape(-1).astype(np.float64)
                nzf = vf[vf != 0]
                if nzf.size > 0.9 * vf.size and np.isfinite(nzf).all():
                    a = np.abs(nzf)
                    # only plausible-looking floats (coordinates written as IEEE floats)
                    if (a > 1e-2).all() and (a < 1e9).all() and np.unique(vf).size > 2:
                        cf = _describe(o + 1, "float32", vf, blocks)
                        # reject floats that are really small integers reinterpreted (denormals etc.)
                        if cf.vmax - cf.vmin > 1e-3:
                            out.append(cf)
    _mark_aliases(out)
    return out


def _mark_aliases(cands: list[Cand]) -> None:
    groups: dict[bytes, list[Cand]] = {}
    for c in cands:
        if c.constant:
            continue
        groups.setdefault(c.values.astype(np.float64).tobytes(), []).append(c)
    for g in groups.values():
        if len(g) > 1:
            for c in g:
                c.aliases = sorted({o.byte for o in g if o is not c})


def find(cands: list[Cand], byte: int, typ: str | None = None) -> Cand | None:
    for c in cands:
        if c.byte == byte and (typ is None or c.type == typ):
            return c
    return None


# --------------------------------------------------------------------------- results
@dataclass
class CoordFit:
    x: Cand
    y: Cand
    coef_x: np.ndarray  # [x0, a, b]: x = x0 + a*k1 + b*k2  (raw, unscaled)
    coef_y: np.ndarray
    rms_x: float
    rms_y: float
    quality: float  # residual / bin size (smaller is better); <0.05 means exact grid
    det_ratio: float


@dataclass
class Geometry:
    kind: str  # "3D post-stack" | "2D post-stack" | "3D pre-stack" | "2D pre-stack" | "shot gathers" | "unknown"
    inline: Cand | None = None
    crossline: Cand | None = None
    cdp: Cand | None = None  # 2D ensemble key
    sp: Cand | None = None
    offset: Cand | None = None
    ensemble: Cand | None = None  # pre-stack ensemble key
    channel: Cand | None = None
    coords: CoordFit | None = None
    coord_kind: str = ""  # "CDP" | "source" | "receiver" | ...
    confidence: str = "none"
    evidence: list[str] = field(default_factory=list)
    alternatives: dict[str, list[dict]] = field(default_factory=dict)
    swapped_il_xl: bool = False
    pair_unique: float = 0.0
    fold: float = 1.0
    line_spacing: float | None = None  # raw coordinate units between successive 2D CDPs


def _pref_rank(byte: int, pref: list[int]) -> int:
    return pref.index(byte) if byte in pref else len(pref) + byte


def _canonical(c: Cand, pref: list[int], cands: list[Cand]) -> Cand:
    """Among identical copies of a field, choose the conventional byte position."""
    if not c.aliases:
        return c
    options = [c] + [x for x in cands if x.byte in c.aliases and np.array_equal(x.values, c.values)]
    options.sort(key=lambda x: (not x.aligned, _pref_rank(x.byte, pref), x.type != "int32"))
    return options[0]


def _key_like(c: Cand) -> bool:
    if c.constant or c.type == "float32":
        return False
    if abs(c.vmax) > 5_000_000 or abs(c.vmin) > 5_000_000:
        return False
    if not c.aligned:
        return False
    return True


def _is_grid(s: np.ndarray, f: np.ndarray, blocks: np.ndarray, f_step: float) -> bool:
    """Tell a 3D grid (inline x crossline) from a 2D line numbered twice (e.g. shotpoint + CDP).

    On a grid the crossline number *resets* when the inline changes (back to the start of the
    next row), and the same crossline numbers recur on many inlines. On a line the second key just
    keeps counting across a change of the first, and no value ever recurs under another key."""
    same_block = blocks[1:] == blocks[:-1]
    change = same_block & (s[1:] != s[:-1])
    if change.any():
        d = np.diff(f)[change]
        step = f_step if f_step else 1.0
        continuing = (np.sign(d) == np.sign(step)) & (np.abs(d) <= 2 * abs(step))
        if continuing.mean() > 0.8:
            return False
        return True
    # no change of the slow key inside any sampled block: fall back to value reuse across blocks
    pairs = {}
    for a, b in zip(s.tolist(), f.tolist()):
        pairs.setdefault(b, set()).add(a)
    reuse = sum(1 for v in pairs.values() if len(v) > 1) / max(len(pairs), 1)
    return reuse > 0.05 or np.unique(s).size <= 2


def _pair_stats(s: np.ndarray, f: np.ndarray) -> float:
    pairs = s.astype(np.int64) * 4_000_003 + f.astype(np.int64)
    return np.unique(pairs).size / max(pairs.size, 1)


def _fit_plane(k1: np.ndarray, k2: np.ndarray, v: np.ndarray):
    A = np.column_stack([np.ones_like(k1, dtype=np.float64), k1.astype(np.float64), k2.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(A, v.astype(np.float64), rcond=None)
    resid = v - A @ coef
    return coef, float(np.sqrt(np.mean(resid**2)))


# Standard fields that hold counters/keys, never coordinates.
_NOT_COORD = {1, 5, 9, 13, 17, 21, 25, 37, 189, 193, 197}


def _coord_candidates(cands: list[Cand], exclude: set[int]) -> list[Cand]:
    out = []
    for c in cands:
        if c.constant or c.byte in exclude or c.type == "int16" or c.byte in _NOT_COORD:
            continue
        std = TRACE_FIELD_BY_BYTE.get(c.byte)
        coordinate_like = np.median(np.abs(c.values[c.values != 0])) > 1e4 if np.any(c.values != 0) else False
        if c.type == "float32":
            coordinate_like = coordinate_like or (np.ptp(c.values) > 1e-3 and c.nunique > 0.2 * c.values.size)
        if std is not None and std.size == 2 and not (coordinate_like and c.nunique > 0.2 * c.values.size):
            continue  # an int32 glued across two standard 2-byte fields
        if c.nunique < min(10, max(3, 0.05 * c.values.size)):  # coordinates vary a lot
            continue
        if c.type == "int32" and not c.aligned:
            continue
        out.append(c)
    return out


def _best_coord_pair(k1: np.ndarray, k2: np.ndarray, step1: float, step2: float, cands: list[Cand], exclude: set[int],
                     scale: np.ndarray | None = None):
    """scale: per-trace coordinate scalar factor (only when byte 71 varies between traces)."""
    fits = []
    for c in _coord_candidates(cands, exclude):
        v = c.values.astype(np.float64)
        if scale is not None and c.type == "int32":
            v = v * scale
        nz = c.values != 0
        if 0.6 < nz.mean() < 1.0 and nz.sum() >= 3:  # some traces have null (0) coordinates: fit the rest
            coef, rms = _fit_plane(k1[nz], k2[nz], v[nz])
        else:
            coef, rms = _fit_plane(k1, k2, v)
        bin_ = max(abs(coef[1] * step1), abs(coef[2] * step2))
        if bin_ <= 0:
            continue
        fits.append((c, coef, rms, rms / bin_))
    good = [f for f in fits if f[3] < 0.6]
    # Within one binary exponent an IEEE float's bit pattern is linear in its value, so the int32 reading of a float
    # coordinate can also fit a plane. When the float reading of the same word fits, it is the real one.
    float_ok = {f[0].byte for f in good if f[0].type == "float32"}
    good = [f for f in good if not (f[0].type == "int32" and f[0].byte in float_ok)]
    best = None
    for i, (cx, ax, rx, qx) in enumerate(good):
        for cy, ay, ry, qy in good:
            if cy is cx or cy.byte == cx.byte or cy.byte in cx.aliases or cy.type != cx.type:
                continue
            J = np.array([[ax[1] * step1, ax[2] * step2], [ay[1] * step1, ay[2] * step2]])
            n1 = np.linalg.norm(J[:, 0])
            n2 = np.linalg.norm(J[:, 1])
            if n1 == 0 or n2 == 0:
                continue
            det_ratio = abs(np.linalg.det(J)) / (n1 * n2)
            if det_ratio < 0.2:
                continue
            aspect = max(n1, n2) / min(n1, n2)  # real bins are roughly square (rarely > 1:8)
            pair = (cx.byte, cy.byte)
            prior = COORD_PAIR_PREFERENCE.index(pair) if pair in COORD_PAIR_PREFERENCE else (10 if cy.byte == cx.byte + 4 else 20)
            q = max(qx, qy)
            # anything better than 1/1000 of a bin is "exact" (coordinates are rounded); beyond that, priors decide
            score = (-np.log10(max(q, 1e-3))) * 3 - prior * 0.15 + det_ratio + (0.5 if cx.type == "int32" else 0) \
                - (2.0 if aspect > 8 else 0.0) - (0.3 if cx.byte > cy.byte else 0.0)
            if best is None or score > best[0]:
                best = (score, CoordFit(cx, cy, ax, ay, rx, ry, q, det_ratio))
    return best[1] if best else None, fits


def infer(cands: list[Cand], blocks: np.ndarray, claims: dict[str, list[int]] | None = None) -> Geometry:
    claims = claims or {}
    m = cands[0].values.size if cands else 0
    # 2-byte readings that are really halves of genuine 4-byte numbers (e.g. coordinates) are not keys
    halves: set[int] = set()
    for c in cands:
        if c.type == "int32" and c.aligned and not c.constant and max(abs(c.vmin), abs(c.vmax)) > 32767:
            halves.update((c.byte, c.byte + 2))
    keys = [c for c in cands if _key_like(c) and not (c.type == "int16" and c.byte in halves)]
    slow = [c for c in keys if c.change_frac < 0.5 and c.nunique >= 2]
    fast = [c for c in keys if c.change_frac >= 0.3 and c.nunique >= 2]

    # ---------------------------------------------------------------- 3D post-stack
    scal = find(cands, 71, "int16")
    scale = None
    if scal is not None and not scal.constant:
        sv = scal.values.astype(np.float64)
        scale = np.where(sv > 0, sv, np.where(sv < 0, 1.0 / np.abs(np.where(sv == 0, 1, sv)), 1.0))
    ranked3d = []
    tried = 0
    for s in slow:
        for f in fast:
            if f.byte == s.byte or f.byte in s.aliases:
                continue
            u = _pair_stats(s.values, f.values)
            if u < 0.3:
                continue
            if not _is_grid(s.values, f.values, blocks, f.step):
                continue  # e.g. a 2D line: shotpoint + CDP are unique together but never form a grid
            tried += 1
            # crossline should reset when inline changes: its range within one inline ~ global range
            same_s = np.r_[False, (s.values[1:] == s.values[:-1]) & (blocks[1:] == blocks[:-1])]
            chg = (~same_s[1:]) & (blocks[1:] == blocks[:-1])
            resets = float(np.mean(np.abs(np.diff(f.values))[chg] > 0)) if chg.any() else 0.5
            small_steps = (abs(s.step) <= 100) + (abs(f.step) <= 100)
            prior = -0.05 * min(_pref_rank(s.byte, INLINE_PREFERENCE), 30) - 0.05 * min(_pref_rank(f.byte, XLINE_PREFERENCE), 30)
            claim_bonus = (1.0 if s.byte in claims.get("inline", []) else 0) + (1.0 if f.byte in claims.get("crossline", []) else 0)
            # a running counter (e.g. trace sequence number) is unique by itself; true crosslines repeat across inlines
            counter_pen = 1.5 if f.nunique > 0.9 * m and m > 50 else 0.0
            score = 6 * min(u, 0.97) + 0.5 * resets + 0.5 * small_steps + prior + claim_bonus - counter_pen + f.step_frac
            ranked3d.append((score, s, f, u))
    ranked3d.sort(key=lambda t: -t[0])
    for _, s, f, u in ranked3d[:6]:
        s = _canonical(s, INLINE_PREFERENCE, cands)
        f = _canonical(f, XLINE_PREFERENCE, cands)
        exclude = {s.byte, f.byte, *s.aliases, *f.aliases}
        cf, fits = _best_coord_pair(s.values, f.values, s.step or 1, f.step or 1, cands, exclude, scale)
        n_s, n_f = s.nunique, f.nunique
        off = find(cands, 37, "int32")
        offset_varies = off is not None and not off.constant and off.byte not in (s.byte, f.byte)
        gather_keys = (s.byte, f.byte) in _GATHER_PAIRS or f.byte == 37 or s.byte == 37
        looks_prestack = offset_varies or (gather_keys and (cf is None or cf.quality > 0.05))
        # pairs that repeat (duplicated / merged volumes) only count as a grid when coordinates prove it
        accept = (cf is not None or (n_s >= 2 and n_f >= 2 and u > 0.995)) if u >= 0.97 else \
            (cf is not None and cf.quality < 0.05)
        if not looks_prestack and accept:
            g = Geometry("3D post-stack", inline=s, crossline=f, coords=cf, pair_unique=u)
            g.evidence.append(
                f"Bytes {s.byte}-{s.byte + s.ref.size - 1} and {f.byte}-{f.byte + f.ref.size - 1} together give a unique "
                f"grid position for {u * 100:.1f}% of {m} sampled traces; the first stays constant for runs of traces "
                f"while the second steps by {f.step:g}.")
            if u < 0.97:
                g.fold = 1 / max(u, 1e-9)
                g.evidence.append(f"About {(1 - u) * 100:.0f}% of sampled traces repeat a grid position already used: the "
                                  "volume contains duplicated traces (e.g. two versions or several stacks concatenated).")
            _add_alias_evidence(g, s, "inline", cands)
            _add_alias_evidence(g, f, "crossline", cands)
            if cf is not None:
                g.evidence.append(
                    f"Coordinates in bytes {cf.x.byte} and {cf.y.byte} are an exact linear function of (inline, crossline): "
                    f"residual {cf.quality * 100:.2f}% of a bin" + (", a perfect regular grid." if cf.quality < 0.05 else "."))
                g.coord_kind = _coord_kind(cf.x.byte)
            g.confidence = "high" if (cf is not None and cf.quality < 0.05 and u > 0.995) else ("medium" if cf is not None or u > 0.995 else "low")
            if u < 0.97:
                g.confidence = "medium"
            g.alternatives["coordinates"] = [
                {"byte": c.byte, "type": c.type, "fit_residual_bins": round(q, 4)} for c, _, _, q in sorted(fits, key=lambda t: t[3])[:6]]
            _apply_claim_naming(g, claims)
            return g

    # ---------------------------------------------------------------- gathers before lines
    c9, c13, c21 = find(cands, 9, "int32"), find(cands, 13, "int32"), find(cands, 21, "int32")
    off = find(cands, 37, "int32")
    offset_varies = off is not None and not off.constant
    runs21 = c21 is not None and not c21.constant and c21.change_frac < 0.75
    runs9 = c9 is not None and not c9.constant and c9.change_frac < 0.75
    cycles13 = c13 is not None and not c13.constant and c13.nunique < 0.5 * m
    if m >= 4 and (offset_varies or runs21 or (runs9 and cycles13)):
        ens = _find_ensembles(cands, keys, blocks)
        if ens is not None:
            return ens

    # ---------------------------------------------------------------- 2D post-stack line
    uniq = [c for c in keys if c.nunique >= 0.97 * m and c.change_frac > 0.9 and c.step_frac > 0.8]
    if uniq and m >= 3:
        pref2d = [21, 193, 1, 5, 17, 197, 9, 25]
        uniq.sort(key=lambda c: (_pref_rank(c.byte, pref2d), c.type != "int32"))
        k = _canonical(uniq[0], pref2d, cands)
        il = find(cands, 189, "int32")
        xl = find(cands, 193, "int32")
        if il is not None and xl is not None and il.constant and il.vmin != 0 and not xl.constant and xl.nunique >= 0.97 * m:
            g = Geometry("3D post-stack", inline=il, crossline=xl, pair_unique=1.0)
            g.evidence.append(f"Bytes 189 (inline) is constant at {il.vmin:g} while 193 (crossline) numbers every trace: "
                              "a single inline of a 3D survey.")
            tmp = Geometry("tmp", cdp=xl)
            _fit_2d(tmp, cands, blocks, {189, 193, *xl.aliases})
            g.coords, g.coord_kind, g.line_spacing = tmp.coords, tmp.coord_kind, tmp.line_spacing
            g.confidence = "medium"
            return g
        g = Geometry("2D post-stack", cdp=k)
        g.evidence.append(f"Bytes {k.byte}-{k.byte + k.ref.size - 1} are unique for every sampled trace and step by {k.step:g}: "
                          "a single line of stacked traces numbered by CDP/trace.")
        _fit_2d(g, cands, blocks, {k.byte, *k.aliases})
        g.sp = _find_shotpoint(cands, k)
        g.confidence = "high" if g.coords is not None else "medium"
        return g

    # ---------------------------------------------------------------- pre-stack gathers
    ens = _find_ensembles(cands, keys, blocks)
    if ens is not None:
        return ens
    g = Geometry("unknown")
    g.evidence.append("No pair of header fields forms a grid, no single field numbers a line, and no ensemble structure was found.")
    return g


_GATHER_PAIRS = {(9, 13), (9, 25), (21, 25), (21, 13), (17, 13), (9, 37), (21, 37)}


def _coord_kind(byte: int) -> str:
    return {181: "CDP", 73: "source", 81: "receiver"}.get(byte, "non-standard")


def _add_alias_evidence(g: Geometry, c: Cand, what: str, cands: list[Cand]) -> None:
    if c.aliases:
        same = [a for a in c.aliases]
        g.evidence.append(f"The {what} number is also stored in byte(s) {', '.join(map(str, same))}; "
                          f"byte {c.byte} was chosen as the conventional location.")


def _apply_claim_naming(g: Geometry, claims: dict[str, list[int]]) -> None:
    """Decide which of the two grid keys is the inline.

    Data alone shows which key changes slowly (sort order), not which one the
    acquisition called 'inline'. Conventional byte positions and textual claims decide.
    """
    s, f = g.inline, g.crossline
    if s is None or f is None:
        return
    s_set = {s.byte, *s.aliases}
    f_set = {f.byte, *f.aliases}
    flip = False
    reason = ""
    if any(b in f_set for b in claims.get("inline", [])) and not any(b in s_set for b in claims.get("inline", [])):
        flip, reason = True, "the textual header names it as the inline"
    elif 189 in f_set and 193 in s_set:
        flip, reason = True, "it sits in byte 189, the SEG-Y rev 1 inline position"
    if flip:
        g.inline, g.crossline = f, s
        g.swapped_il_xl = True
        g.evidence.append(f"The file is sorted by crossline (the slow-changing key is the crossline); the inline was "
                          f"identified as byte {f.byte} because {reason}.")


def _fit_2d(g: Geometry, cands: list[Cand], blocks: np.ndarray, exclude: set[int]) -> None:
    k = g.cdp.values.astype(np.float64)
    same = blocks[1:] == blocks[:-1]
    best = None
    coords = _coord_candidates(cands, exclude)
    floats = {c.byte for c in coords if c.type == "float32"}
    coords = [c for c in coords if not (c.type == "int32" and c.byte in floats)]
    for i, cx in enumerate(coords):
        for cy in coords:
            if cy.byte <= cx.byte and not (cx.byte, cy.byte) in COORD_PAIR_PREFERENCE:
                continue
            if cy is cx or cy.byte in cx.aliases or cy.type != cx.type:
                continue
            dx = np.diff(cx.values.astype(np.float64))[same]
            dy = np.diff(cy.values.astype(np.float64))[same]
            dk = np.diff(k)[same]
            if dk.size < 2:
                continue
            ok = dk != 0
            dist = np.hypot(dx, dy)[ok] / np.abs(dk[ok])
            if dist.size < 2 or np.median(dist) == 0:
                continue
            med = float(np.median(dist))
            # along a line, spacing per CDP is nearly constant
            cv = float(np.median(np.abs(dist - med)) / med)
            if cv > 0.25:
                continue
            pair = (cx.byte, cy.byte)
            prior = COORD_PAIR_PREFERENCE.index(pair) if pair in COORD_PAIR_PREFERENCE else (10 if cy.byte == cx.byte + 4 else 20)
            score = -cv * 10 - prior * 0.2
            if best is None or score > best[0]:
                best = (score, cx, cy, med, cv)
    if best:
        _, cx, cy, med, cv = best
        cfx, rx = np.polyfit(k, cx.values.astype(np.float64), 1), 0.0
        cfy = np.polyfit(k, cy.values.astype(np.float64), 1)
        g.coords = CoordFit(cx, cy, np.array([cfx[1], cfx[0], 0.0]), np.array([cfy[1], cfy[0], 0.0]), 0, 0, cv, 1.0)
        g.coord_kind = _coord_kind(cx.byte)
        g.line_spacing = med
        g.evidence.append(f"Bytes {cx.byte} and {cy.byte} move smoothly along the line: {med:g} raw units per CDP step, "
                          f"varying by only {cv * 100:.1f}%.")


def _find_shotpoint(cands: list[Cand], k: Cand) -> Cand | None:
    for b in (197, 17):
        c = find(cands, b, "int32")
        if c is not None and not c.constant and c.byte not in k.aliases and c.byte != k.byte:
            r = np.corrcoef(c.values.astype(float), k.values.astype(float))[0, 1] if c.values.std() > 0 else 0
            if abs(r) > 0.95:
                return c
    return None


def _find_ensembles(cands: list[Cand], keys: list[Cand], blocks: np.ndarray) -> Geometry | None:
    m = cands[0].values.size
    # an ensemble key repeats for runs (change_frac small) and a within-ensemble key cycles
    ens_pref = [21, 9, 189, 17, 5]
    runs = [c for c in keys if 0 < c.change_frac < 0.7 and c.nunique >= 2]
    runs.sort(key=lambda c: (_pref_rank(c.byte, ens_pref), c.change_frac))
    if not runs:
        return None
    # 3D pre-stack: (IL, XL) pairs repeat
    il = find(cands, 189, "int32")
    xl = find(cands, 193, "int32")
    off = find(cands, 37, "int32")
    if il is not None and xl is not None and not il.constant and not xl.constant:
        u = _pair_stats(il.values, xl.values)
        if u < 0.9:
            g = Geometry("3D pre-stack", inline=il, crossline=xl, offset=off if off is not None and not off.constant else None,
                         pair_unique=u)
            g.fold = 1 / max(u, 1e-9)
            g.evidence.append(f"Inline/crossline pairs (bytes 189/193) repeat about {g.fold:.0f} times each: "
                              "these are pre-stack gathers, not a stacked volume.")
            g.confidence = "medium"
            _gather_coords(g, cands, blocks)
            return g
    e = _canonical(runs[0], ens_pref, cands)
    within = None
    for b in (13, 25, 37, 1):
        c = find(cands, b, "int32")
        if c is not None and not c.constant and c.byte != e.byte and c.byte not in e.aliases:
            within = c
            break
    kind = {9: "shot gathers", 21: "2D pre-stack"}.get(e.byte, "pre-stack gathers")
    runlen = 1.0 / max(e.change_frac, 1e-9)
    g = Geometry(kind, ensemble=e, channel=within, offset=off if off is not None and not off.constant else None)
    g.fold = runlen
    name = TRACE_FIELD_BY_BYTE.get(e.byte).name if TRACE_FIELD_BY_BYTE.get(e.byte) else f"byte {e.byte}"
    g.evidence.append(f"Byte {e.byte} ({name}) stays constant for runs of about {runlen:.1f} traces while "
                      + (f"byte {within.byte} cycles within each run" if within else "other fields vary")
                      + f": the file is organised as ensembles (gathers) of ~{runlen:.0f} traces.")
    g.confidence = "medium"
    if e.aliases:
        g.evidence.append(f"The ensemble number is also stored in byte(s) {', '.join(map(str, e.aliases))}.")
    _gather_coords(g, cands, blocks)
    return g


def _gather_coords(g: Geometry, cands: list[Cand], blocks: np.ndarray) -> None:
    """Pre-stack data: report the first standard coordinate pair that actually varies (CDP, source, receiver)."""
    for bx, by in ((181, 185), (73, 77), (81, 85)):
        cx, cy = find(cands, bx, "int32"), find(cands, by, "int32")
        if cx is None or cy is None or cx.constant or cy.constant:
            continue
        g.coords = CoordFit(cx, cy, np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 1.0)
        g.coord_kind = _coord_kind(bx)
        g.evidence.append(f"{g.coord_kind.capitalize()} coordinates vary in bytes {bx}/{by}.")
        return


def describe_header_map(cands: list[Cand], geom: Geometry) -> list[dict]:
    """Human-readable map of every non-empty field, for the report's 'what's in the headers' table."""
    roles: dict[int, str] = {}
    for role, c in (("inline", geom.inline), ("crossline", geom.crossline), ("CDP / trace key", geom.cdp),
                    ("shotpoint", geom.sp), ("offset", geom.offset), ("ensemble key", geom.ensemble), ("channel", geom.channel)):
        if c is not None:
            roles[c.byte] = role
            for a in c.aliases:
                roles.setdefault(a, f"copy of {role}")
    if geom.coords is not None:
        roles[geom.coords.x.byte] = "X coordinate"
        roles[geom.coords.y.byte] = "Y coordinate"
        for a in geom.coords.x.aliases:
            roles.setdefault(a, "copy of X coordinate")
        for a in geom.coords.y.aliases:
            roles.setdefault(a, "copy of Y coordinate")
    rows = []
    covered: set[int] = set()
    # prefer standard field interpretations, then non-standard int32, then floats
    ordered = sorted(cands, key=lambda c: (c.byte, c.std_name is None, c.type != "int32"))
    for c in ordered:
        if c.byte in covered:
            continue
        if c.constant and c.vmin == 0:
            continue
        std = trace_field_at(c.byte)
        is_std = c.std_name is not None
        if not is_std and c.byte not in roles:
            # skip straddling / sub-field interpretations unless they carry a recognised role
            if c.type != "int32" or not c.aligned or (std and std.byte == c.byte):
                continue
            if std is not None and std.size == 2:
                continue
        covered.update(range(c.byte, c.byte + c.ref.size))
        rows.append({
            "byte": c.byte, "end": c.byte + c.ref.size - 1, "type": c.type,
            "standard": f"{std.name}" if std and std.byte == c.byte else (f"inside {std.name}" if std else "unassigned"),
            "role": roles.get(c.byte, ""),
            "min": c.vmin, "max": c.vmax, "nunique": c.nunique, "constant": c.constant,
            "pattern": _pattern(c),
        })
    return rows


def _pattern(c: Cand) -> str:
    if c.constant:
        return f"constant {c.vmin:g}"
    if c.change_frac < 0.2:
        return f"changes every ~{1 / max(c.change_frac, 1e-9):.0f} traces"
    if c.step_frac > 0.8:
        return f"steps by {c.step:g}"
    return "varies"
