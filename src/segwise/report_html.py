"""Render self-contained HTML reports (single offline files: inline CSS, JS and images)."""
from __future__ import annotations

import base64
import datetime as _dt
import functools
import json
from importlib.resources import files

from jinja2 import Environment, PackageLoader, select_autoescape

from . import __version__


@functools.lru_cache(maxsize=1)
def env() -> Environment:
    e = Environment(loader=PackageLoader("segwise", "templates"), autoescape=select_autoescape(["html"]),
                    trim_blocks=True, lstrip_blocks=True)
    return e


@functools.lru_cache(maxsize=1)
def asset(name: str) -> str:
    return files("segwise").joinpath(f"templates/{name}").read_text()


def css() -> str:
    return asset("style.css")


def js() -> str:
    return asset("app.js")


def figures(report, section: bool = True) -> dict:
    from . import plots

    out = {}
    for key, fn in (("location", plots.location_map), ("plan", plots.plan_view), ("tracks", plots.header_tracks)):
        try:
            out[key] = fn(report)
        except Exception as e:  # a failed figure must never break a report
            report.notes.append(f"Could not draw the {key} figure: {e}")
    if section and report.status != "unusable":
        try:
            out["section"] = plots.section(report)
        except Exception as e:
            report.notes.append(f"Could not draw the section preview: {e}")
    return out


def context(report, fixlog=None, back_link: str | None = None, figs: dict | None = None, embed_loadsheet: bool = True):
    from .exports import load_instructions, loadsheet

    load = load_instructions(report, False) if report.status != "unusable" and report.geometry else None
    load_fixed = load_instructions(report, True) if fixlog is not None else None
    href = None
    if embed_loadsheet and report.status != "unusable":
        data = json.dumps(loadsheet(report, fixlog), indent=2).encode()
        href = "data:application/json;base64," + base64.b64encode(data).decode()
    return {"r": report, "fixlog": fixlog, "figs": figs if figs is not None else figures(report), "load": load,
            "load_fixed": load_fixed, "back_link": back_link, "loadsheet_href": href}


def render_report(report, fixlog=None, back_link: str | None = None) -> str:
    ctx = context(report, fixlog, back_link)
    ctx["back_label"] = "Back to the delivery" if back_link else None
    return env().get_template("report.html").render(css=css(), js=js(), **ctx)


def render_inventory(reports, failures, source: str, groups: dict) -> str:
    from .exports import inventory_row

    rows = [(inventory_row(r), stem) for r, stem in reports]
    order = {"unusable": 0, "needs_review": 1, "fixable": 2, "ready": 3}
    rows.sort(key=lambda t: (order.get(t[0]["status"], 9), t[0]["file_name"]))
    counts: dict[str, int] = {}
    for row, _ in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if failures:
        counts["unusable"] = counts.get("unusable", 0) + len(failures)
    attention = [t for t in rows if t[0]["status"] in ("needs_review", "unusable")][:60]
    try:
        overview = overview_map([r for r, _ in reports])
    except Exception:
        overview = None
    return env().get_template("inventory.html").render(
        css=css(), js=js(), rows=rows, failures=failures, source=source, groups=groups, counts=counts,
        attention=attention, overview=overview, version=__version__,
        generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"))


def overview_map(reports) -> str | None:
    import numpy as np

    from . import plots
    from .crs import land_rings

    polys = [np.asarray(r.plot.get("outline_lonlat"), float) for r in reports if r.plot.get("outline_lonlat")]
    if not polys:
        return None
    allp = np.concatenate(polys)
    lon0, lat0 = allp.min(axis=0)
    lon1, lat1 = allp.max(axis=0)
    span = max(lon1 - lon0, lat1 - lat0, 0.05)
    pad = max(span * 0.35, 1.5)
    fig, ax = plots.plt.subplots(figsize=(9, 3.8))
    ax.set_facecolor(plots.SEA)
    for ring in land_rings():
        if ring[:, 0].max() < lon0 - pad or ring[:, 0].min() > lon1 + pad or ring[:, 1].max() < lat0 - pad or ring[:, 1].min() > lat1 + pad:
            continue
        ax.fill(ring[:, 0], ring[:, 1], color=plots.LAND, ec=plots.LAND_EDGE, lw=.5)
    colors = {"ready": "#1d7f55", "fixable": "#2c5bc9", "needs_review": "#c27a12", "unusable": "#b4302a"}
    for r, p in zip([r for r in reports if r.plot.get("outline_lonlat")], polys):
        ax.fill(p[:, 0], p[:, 1], color=colors.get(r.status, plots.INK), alpha=.18)
        ax.plot(p[:, 0], p[:, 1], color=colors.get(r.status, plots.INK), lw=1.4)
        ax.annotate(r.file_name, (p[:, 0].mean(), p[:, 1].max()), textcoords="offset points", xytext=(0, 4),
                    fontsize=6.5, color=plots.INK2, ha="center")
    ax.set_xlim(lon0 - pad, lon1 + pad)
    ax.set_ylim(lat0 - pad, lat1 + pad)
    ax.set_aspect(1 / max(np.cos(np.radians((lat0 + lat1) / 2)), .2), adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.grid(color="white", lw=.8)
    for s in ax.spines.values():
        s.set_visible(False)
    return plots._png(fig)
