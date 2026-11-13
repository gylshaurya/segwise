"""Figures for reports (matplotlib, offline, returned as base64 PNG data URIs).

  location_map   where on Earth the survey lands (offline coastlines) - the
                 single best sanity check for CRS and scalar mistakes
  plan_view      trace positions in map coordinates with corner labels
  section        a seismic section preview (middle inline / the 2D line / a few gathers)
  header_tracks  key header fields plotted against trace number, like log tracks:
                 resets, jumps and gaps in geometry are visible at a glance
"""
from __future__ import annotations

import base64
import io

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)  # aspect-lock notices, not errors
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#16181c"
INK2 = "#4b5059"
INK3 = "#6b7078"
LINE = "#e7e7e3"
SEA = "#eef2f5"
LAND = "#e6e1d8"
LAND_EDGE = "#cfc7b9"
ACCENT = "#e0764f"
ACCENT2 = "#5a6fd8"
SEIS = LinearSegmentedColormap.from_list("seis", ["#2c4a9a", "#9fb2df", "#f7f7f5", "#e3a08a", "#a7321f"])

plt.rcParams.update({
    "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK3, "ytick.color": INK3,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold", "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def location_map(report) -> str | None:
    ll = report.plot.get("lonlat")
    if ll is None or len(ll) == 0:
        return None
    from .crs import land_rings

    ll = np.asarray(ll, float)
    ok = np.isfinite(ll).all(axis=1)
    ll = ll[ok]
    if not len(ll):
        return None
    # surveys across the 180 degree meridian (e.g. Bering Sea): work in 0..360 longitudes
    wrap = np.ptp(ll[:, 0]) > 180
    if wrap:
        ll = ll.copy()
        ll[:, 0] = np.where(ll[:, 0] < 0, ll[:, 0] + 360, ll[:, 0])
    lon0, lon1 = ll[:, 0].min(), ll[:, 0].max()
    lat0, lat1 = ll[:, 1].min(), ll[:, 1].max()
    span = max(lon1 - lon0, lat1 - lat0, 0.02)
    pad = max(span * 3.0, 2.2)
    cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_facecolor(SEA)
    rings = land_rings()
    if wrap:
        rings = list(rings) + [r + np.array([360.0, 0.0], np.float32) for r in rings]
    box = (cx - pad * 1.5, cy - pad, cx + pad * 1.5, cy + pad)
    for r in rings:
        if r[:, 0].max() < box[0] or r[:, 0].min() > box[2] or r[:, 1].max() < box[1] or r[:, 1].min() > box[3]:
            continue
        ax.fill(r[:, 0], r[:, 1], color=LAND, ec=LAND_EDGE, lw=0.5, zorder=1)
    ol = report.plot.get("outline_lonlat")
    if ol:
        o = np.asarray(ol, float)
        if wrap:
            o[:, 0] = np.where(o[:, 0] < 0, o[:, 0] + 360, o[:, 0])
        closed = len(o) > 3 and np.allclose(o[0], o[-1])
        if closed:
            ax.fill(o[:, 0], o[:, 1], color=ACCENT, alpha=.25, zorder=3)
        ax.plot(o[:, 0], o[:, 1], color=ACCENT, lw=1.8, zorder=4)
    ax.scatter(ll[::max(1, len(ll) // 600), 0], ll[::max(1, len(ll) // 600), 1], s=2, color=ACCENT, zorder=4)
    ax.plot([cx], [cy], marker="o", ms=11, mfc="none", mec=INK, mew=1.4, zorder=5)
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect(1 / max(np.cos(np.radians(cy)), 0.2), adjustable="datalim")
    ax.grid(color="white", lw=0.8, zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    for s in ax.spines.values():
        s.set_visible(False)
    # world locator
    ins = ax.inset_axes([0.70, 0.72, 0.28, 0.25])
    ins.set_facecolor(SEA)
    for r in land_rings():
        if len(r) > 30:
            ins.fill(r[::4, 0], r[::4, 1], color=LAND, lw=0)
    ins.plot([cx - 360 if cx > 180 else cx], [cy], "o", ms=5, color=ACCENT, mec="white", mew=1)
    ins.set_xlim(-180, 180)
    ins.set_ylim(-70, 85)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(LINE)
    return _png(fig)


def plan_view(report) -> str | None:
    x = report.plot.get("x")
    y = report.plot.get("y")
    if x is None or len(x) < 2:
        return None
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    G = report.geometry
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    c = report.plot.get("il")
    if c is not None and len(c) == len(x):
        sc = ax.scatter(x, y, c=np.asarray(c), s=3, cmap=LinearSegmentedColormap.from_list("p", ["#b9c4f2", ACCENT2, "#1d2a5a"]), lw=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("inline")
        cb.outline.set_visible(False)
    else:
        ax.plot(x, y, color=ACCENT2, lw=1.4)
        ax.scatter(x[[0, -1]], y[[0, -1]], color=[ACCENT, INK], s=24, zorder=3)
        ax.annotate("first trace", (x[0], y[0]), textcoords="offset points", xytext=(6, 6), color=INK2, fontsize=8)
    for k in G.get("corners") or []:
        ax.plot(k["x"], k["y"], "o", ms=5, color=INK)
        ax.annotate(f"IL {k['inline']}\nXL {k['crossline']}", (k["x"], k["y"]), textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5, color=INK2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("X (easting)")
    ax.set_ylabel("Y (northing)")
    ax.grid(color=LINE, lw=0.6)
    return _png(fig)


def section(report, max_traces: int = 1200) -> tuple[str, str] | None:
    from .reader import TraceReader, probe

    try:
        lay = probe(report.path)
    except Exception:
        return None
    if lay.n_traces == 0:
        return None
    r = TraceReader(lay)
    G = report.geometry
    fmt = report.samples.get("format", lay.format_code)
    caption = ""
    idx = None
    if G.get("inline") and G.get("il_min") is not None:
        il_ref = G["inline"]
        mid = (G["il_min"] + G["il_max"]) // 2
        inc = G.get("il_inc") or 1
        mid = G["il_min"] + ((mid - G["il_min"]) // inc) * inc
        n = lay.n_traces
        # find traces of that inline cheaply: sample to locate, then read a contiguous window
        step = max(1, n // 20000)
        probe_idx = np.arange(0, n, step)
        ilv = r.field(probe_idx, il_ref["byte"], il_ref["type"])
        hit = probe_idx[ilv == mid]
        if hit.size:
            lo = max(0, int(hit.min()) - step)
            hi = min(n, int(hit.max()) + step + 1)
            window = np.arange(lo, hi)
            wv = r.field(window, il_ref["byte"], il_ref["type"])
            idx = window[wv == mid]
            caption = f"Inline {mid} ({idx.size} traces)"
    if idx is None or idx.size < 2:
        n = lay.n_traces
        idx = np.arange(0, min(n, max_traces)) if n <= max_traces else np.unique(np.linspace(0, n - 1, max_traces).astype(int))
        caption = f"Traces {idx[0] + 1}-{idx[-1] + 1}" + ("" if n <= max_traces else f" (every {max(1, n // max_traces)}th)")
    if idx.size > max_traces:
        idx = idx[np.linspace(0, idx.size - 1, max_traces).astype(int)]
    d = r.samples(idx, fmt).astype(np.float64).T
    d[~np.isfinite(d)] = 0
    big = np.abs(d) > 1e30
    d[big] = 0
    # display-only AGC (as on any QC plot): divide by a moving RMS so deep, weak events are visible
    win = max(5, int(250_000 / max(lay.dt_us or 4000, 1)))  # ~250 ms window
    if d.shape[0] > 2 * win:
        k = np.ones(win) / win
        env_ = np.sqrt(np.apply_along_axis(lambda t: np.convolve(t * t, k, mode="same"), 0, d)) + 1e-30
        d = d / env_
        caption_gain = "; display gain (AGC) applied"
    else:
        caption_gain = ""
    a = np.abs(d[d != 0])
    clip = np.percentile(a, 98) if a.size else 1.0
    clip = clip or 1.0
    ns, nt = d.shape
    dt = (lay.dt_us or 1000) / 1000.0
    depth = (report.text or {}).get("domain") == "depth"
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.imshow(d, aspect="auto", cmap=SEIS, vmin=-clip, vmax=clip, extent=[0, nt, ns * dt, 0], interpolation="bilinear")
    ax.set_xlabel("trace (in display order)")
    ax.set_ylabel("depth" if depth else "time (ms)")
    for s in ax.spines.values():
        s.set_visible(False)
    return _png(fig), caption + caption_gain


def header_tracks(report) -> str | None:
    P = report.plot
    t = P.get("trace")
    if t is None or len(t) < 3:
        return None
    t = np.asarray(t)
    tracks = []
    if P.get("il") is not None:
        tracks += [("Inline", P["il"], "#2f8a5b"), ("Crossline", P["xl"], "#c0392b")]
    if P.get("x") is not None:
        tracks += [("X", P["x"], ACCENT2), ("Y", P["y"], "#8a5cc8")]
    rms_i, rms = P.get("rms_idx"), P.get("rms")
    if rms_i is not None and len(rms_i) > 2:
        tracks.append(("RMS amplitude", (np.asarray(rms_i), np.asarray(rms)), INK))
    if not tracks:
        return None
    fig, axes = plt.subplots(1, len(tracks), figsize=(1.55 * len(tracks) + 0.6, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, vals, col) in zip(axes, tracks):
        if isinstance(vals, tuple):
            ty, v = vals
        else:
            ty, v = t, np.asarray(vals, float)
        ax.plot(v, ty, color=col, lw=0.9)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(color=LINE, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        for s in ("left", "bottom"):
            ax.spines[s].set_color(LINE)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("trace number")
    fig.tight_layout(w_pad=0.6)
    return _png(fig)
