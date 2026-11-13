"""Real public data: a cropped F3 Netherlands cube and a 1986 USGS Bering Sea survey."""
import glob
import os
import re

import numpy as np
import pytest

from segwise.analyze import analyze
from segwise.writer import write_fixed
from conftest import NAMSS, SEGYIO, needs_namss, needs_segyio_data


@needs_segyio_data
def test_f3_geometry_and_location():
    r = analyze(f"{SEGYIO}/f3.sgy")
    G = r.geometry
    assert G["kind"] == "3D post-stack"
    assert (G["il_min"], G["il_max"], G["xl_min"], G["xl_max"]) == (111, 133, 875, 892)
    assert G["inline_spacing"] == pytest.approx(25, abs=0.05) and G["crossline_spacing"] == pytest.approx(25, abs=0.05)
    assert (G["x"]["byte"], G["y"]["byte"], G["scalar_effective"]) == (181, 185, -10)
    # "Dutch North Sea" in the text + coordinates -> UTM zone 31N; datum honestly left open
    assert r.crs["status"] == "ambiguous" and r.crs["candidates"][0]["epsg"] in (23031, 32631, 25831)
    lon, lat = G["centre_lonlat"]
    assert 4.5 < lon < 5.2 and 54.6 < lat < 55.0 and G["centre_on_land"] is False
    assert {"inline", 9} and 9 in [a["byte"] for a in r.finding("inline").alternatives]  # also stored in byte 9


@needs_segyio_data
def test_f3_fix_loads_in_segyio(tmp_path):
    import segyio

    r = analyze(f"{SEGYIO}/f3.sgy", overrides={"epsg": 23031})
    log = write_fixed(r, str(tmp_path / "f3_std.sgy"))
    assert log.verification["passed"]
    with segyio.open(log.output) as f:  # default byte locations
        assert list(f.ilines) == list(range(111, 134)) and list(f.xlines) == list(range(875, 893))
        with segyio.open(f"{SEGYIO}/f3.sgy") as g:
            assert np.array_equal(segyio.tools.cube(f), segyio.tools.cube(g))


@needs_namss
def test_namss_lines_have_no_coordinates():
    r = analyze(f"{NAMSS}/Data/SEGY/Sort/f2860002.sgy")
    assert r.geometry["kind"] == "2D pre-stack"
    assert "NO_COORDINATES" in {i.code for i in r.issues}
    assert r.samples["format"] == 1 and r.layout["ns"] == 2500


@needs_namss
def test_namss_navigation_matches_every_line():
    from segwise.nav import match, read_nav

    nav = read_nav(f"{NAMSS}/Navigation/f-2-86-bs.410")
    files = sorted(glob.glob(f"{NAMSS}/Data/SEGY/Sort/*.sgy"))
    assert len(files) == 39
    for p in files:
        m = match(analyze(p, full_scan=False), nav)
        expect = "F286-" + str(int(re.search(r"(\d{2})\.sgy$", p).group(1)))
        assert m.line == expect, (os.path.basename(p), m.line)


@needs_namss
def test_namss_merge_produces_positioned_line(tmp_path):
    from segwise.nav import match, read_nav

    p = f"{NAMSS}/Data/SEGY/Sort/f2860017.sgy"
    r = analyze(p)
    m = match(r, read_nav(f"{NAMSS}/Navigation/f-2-86-bs.410"))
    log = write_fixed(r, str(tmp_path / "l17.sgy"), nav=m.as_writer_nav())
    assert log.verification["passed"]
    r2 = analyze(log.output, overrides={"epsg": m.to_epsg})
    lon, lat = r2.geometry["centre_lonlat"]
    assert 170 < lon or lon < -160  # Bering Sea straddles the antimeridian
    assert 54 < lat < 64 and r2.geometry["centre_on_land"] is False
