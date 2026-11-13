"""Check the declared sample format against the samples themselves, and QC amplitudes.

A seismic trace is a band-limited wiggle: neighbouring samples are strongly
correlated, values are finite, and their magnitudes span a few orders of
magnitude. Decoding the bytes with the *wrong* number format (the classic
case: IBM floats read as IEEE floats) produces values that are wildly
erratic, astronomically large or tiny, or NaN. We decode a sample of traces
with every format of the same byte width and score how "seismic-like" each
result is. The declared format wins unless it is clearly implausible and an
alternative is clearly plausible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .formats import FORMATS, SAME_WIDTH, format_name
from .reader import Layout, TraceReader


@dataclass
class FormatVerdict:
    declared: int
    effective: int
    scores: dict[int, float]
    evidence: list[str] = field(default_factory=list)
    changed: bool = False
    confidence: str = "high"


def plausibility(x: np.ndarray) -> tuple[float, dict]:
    """Score in [0, 1]: how much does a (k, ns) array look like seismic traces?"""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    ff = float(finite.mean()) if x.size else 0.0
    x = np.where(finite, x, 0.0)
    a = np.abs(x)
    nz = a[a > 0]
    info = {"finite": ff}
    if nz.size < max(10, 0.01 * x.size):
        info["dead"] = True
        return 0.5, info  # all zero: tells us nothing about the format
    lg = np.log10(nz)
    spread = float(np.percentile(lg, 97) - np.percentile(lg, 3))
    extreme = float(np.mean((nz > 1e20) | (nz < 1e-20)))
    live = a.max(axis=1) > 0
    xs = x[live]
    if xs.shape[1] > 3:
        xc = xs - xs.mean(axis=1, keepdims=True)
        num = (xc[:, :-1] * xc[:, 1:]).sum(axis=1)
        den = np.sqrt((xc[:, :-1] ** 2).sum(axis=1) * (xc[:, 1:] ** 2).sum(axis=1)) + 1e-300
        ac = float(np.median(num / den))
    else:
        ac = 0.0
    # magnitude jumps between neighbours (garbage formats jump by many orders of magnitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.log10(np.where(xs != 0, np.abs(xs), np.nan))
    jumps = np.abs(np.diff(la, axis=1))
    jump = float(np.nanmedian(jumps)) if np.isfinite(jumps).any() else 0.0
    info.update(spread=spread, extreme=extreme, autocorr=ac, jump=jump)
    s = ff * (1 - extreme)
    s *= float(np.clip((ac + 0.3) / 0.9, 0.05, 1.0))
    if spread > 10:
        s *= 0.2
    elif spread < 0.15:  # every value has (almost) the same magnitude: typical of IBM bits read as integers
        s *= 0.3
    if jump > 1.5:
        s *= 0.3
    return float(np.clip(s, 0, 1)), info


def verify_format(reader: TraceReader, lay: Layout, n: int = 200) -> FormatVerdict:
    declared = lay.format_code
    width = FORMATS[declared][0]
    if lay.n_traces == 0:
        return FormatVerdict(declared, declared, {}, ["No traces to check."], confidence="none")
    idx = np.unique(np.linspace(0, lay.n_traces - 1, min(n, lay.n_traces)).astype(np.int64))
    alts = [c for c in SAME_WIDTH.get(width, [declared]) if c in (1, 2, 3, 5, 6, 8, 9, 7)] or [declared]
    if declared not in alts:
        alts.insert(0, declared)
    scores, infos = {}, {}
    for c in alts:
        try:
            smp = reader.samples(idx, c)
        except Exception:
            continue
        scores[c], infos[c] = plausibility(smp)
    v = FormatVerdict(declared, declared, scores)
    d = scores.get(declared, 0.0)
    if width == 4:
        sig = ibm_signature(reader, lay, idx)
        if sig is not None:
            zfrac, nwords = sig
            v.evidence.append(
                f"IBM-normalisation test on {nwords:,} non-zero samples: {zfrac * 100:.2f}% have a zero leading fraction "
                "digit (genuine IBM floats: 0%; IEEE floats: typically 5-10%).")
            if declared == 1 and zfrac > 0.02 and scores.get(5, 0) > 0.3:
                v.effective, v.changed = 5, True
                v.evidence.append("The samples are declared IBM but cannot be normalised IBM numbers: they are IEEE floats. "
                                  "Loading them as IBM would distort every amplitude.")
                return v
            if declared in (5, 2, 4) and zfrac < 0.0005 and nwords >= 2000 and scores.get(1, 0) > 0.3:
                v.effective, v.changed = 1, True
                v.evidence.append(f"The samples are declared format {declared} ({format_name(declared)}) but every one of "
                                  f"{nwords:,} values is a perfectly normalised IBM float, which would essentially never "
                                  "happen by chance. They are IBM floats; reading them otherwise distorts every amplitude.")
                return v
            if declared in (1, 5):
                v.evidence.append("This agrees with the declared format.")
                return v
    if infos.get(declared, {}).get("dead"):
        v.evidence.append("The sampled traces are all zero, so the sample format cannot be verified from the data.")
        v.confidence = "low"
        return v
    best = max(scores, key=scores.get) if scores else declared
    if best != declared and scores[best] > 0.5 and scores[best] > 2.5 * max(d, 0.05):
        v.effective = best
        v.changed = True
        v.evidence.append(
            f"Decoding the samples as format {declared} ({format_name(declared)}) gives implausible values "
            f"(score {d:.2f}); decoding them as format {best} ({format_name(best)}) gives smooth, well-behaved "
            f"seismic traces (score {scores[best]:.2f}).")
    else:
        v.evidence.append(f"Samples decoded as format {declared} ({format_name(declared)}) look like seismic data "
                          f"(plausibility {d:.2f}" + (", best of all candidates)." if best == declared else
                                                         f"; format {best} scored {scores[best]:.2f} but not decisively better)."))
        if d < 0.3:
            v.confidence = "low"
            v.evidence.append("The plausibility is low for every format: the traces may be noise, heavily clipped or encoded "
                              "in an unusual way.")
    return v


def ibm_signature(reader: TraceReader, lay: Layout, idx: np.ndarray):
    raw = reader.raw_samples(idx, 1 if FORMATS[lay.format_code][0] == 4 else lay.format_code)
    w = np.ascontiguousarray(raw).view(np.dtype(lay.endian + "u4")).ravel()
    w = w[(w & np.uint32(0x7FFFFFFF)) != 0]
    if w.size < 500:
        return None
    nib = (w >> np.uint32(20)) & np.uint32(0xF)
    return float(np.mean(nib == 0)), int(w.size)


@dataclass
class AmpStats:
    n_checked: int
    dead_frac: float
    nan_frac: float
    inf_frac: float
    clip_frac: float
    p01: float
    p50: float
    p99: float
    absmax: float
    rms: float
    trace_rms: np.ndarray  # per checked trace
    checked_idx: np.ndarray
    constant_traces: int = 0
    spikes: int = 0
    spike_max: float = 0.0
    ibm_overflow: int = 0

    def to_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in self.__dict__.items()
                if k not in ("trace_rms", "checked_idx")}


def amplitude_stats(reader: TraceReader, lay: Layout, fmt: int, n: int = 1500) -> AmpStats:
    if lay.n_traces == 0:
        z = np.zeros(0)
        return AmpStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, z, z.astype(int))
    idx = np.unique(np.linspace(0, lay.n_traces - 1, min(n, lay.n_traces)).astype(np.int64))
    parts, nan_n, overflow_n = [], 0, 0
    for s in range(0, idx.size, 256):
        raw = reader.raw_samples(idx[s : s + 256], fmt)
        x = reader.samples(idx[s : s + 256], fmt).astype(np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            nan_n += int(np.isnan(x).sum())
            if FORMATS[fmt][1] == "ibm":
                overflow_n += int(np.isinf(x).sum())  # valid IBM numbers beyond the IEEE single range
            else:
                nan_n += 0
        parts.append(x)
    x = np.concatenate(parts) if parts else np.zeros((0, lay.ns))
    nan = np.isnan(x)
    inf = np.isinf(x)
    fin = np.where(nan | inf, 0.0, x)
    a = np.abs(fin)
    trace_max = a.max(axis=1) if a.size else np.zeros(0)
    dead = trace_max == 0
    const = int(np.sum((fin.max(axis=1) == fin.min(axis=1)) & ~dead)) if fin.size else 0
    live = fin[~dead]
    st = AmpStats(int(idx.size), float(dead.mean()) if dead.size else 0.0, float(nan.mean()) if x.size else 0.0,
                  float(inf.mean()) if x.size else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros(0), idx, const)
    st.ibm_overflow = overflow_n
    if live.size:
        al = np.abs(live)
        nzv = al[al > 0]
        p01, p50, p99 = (float(np.percentile(nzv, q)) for q in (1, 50, 99))
        spikes = al > max(p99, 1e-30) * 1e4
        st.spikes = int(spikes.sum())
        st.spike_max = float(al.max()) if st.spikes else 0.0
        clean = np.where(spikes, 0.0, live)
        ac = np.abs(clean)
        am = float(ac.max())
        # clipping = flat tops: runs of consecutive samples sitting exactly at the extreme value
        top = ac >= am * 0.99999 if am > 0 else np.zeros_like(ac, bool)
        runs = int(np.sum(top[:, 1:] & top[:, :-1]))
        st.clip_frac = float(top.mean()) if (runs >= 10 and top.mean() > 0.001) else 0.0
        st.p01, st.p50, st.p99, st.absmax = p01, p50, p99, am
        st.rms = float(np.sqrt(np.mean(clean**2)))
        st.trace_rms = np.sqrt(np.mean(np.where(np.abs(fin) > max(p99, 1e-30) * 1e4, 0, fin) ** 2, axis=1))
    else:
        st.trace_rms = np.zeros(fin.shape[0])
    return st
