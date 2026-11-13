import numpy as np

from segwise import crs
from segwise.textheader import parse


def test_byte_claims_and_ranges():
    t = parse(["C 1 TRACE HEADER BYTE LOCATIONS:", "C 2 INLINE: 9-12  XLINE: 21-24  CDP X: 73-76  CDP Y: 77-80",
               "C 3 INLINE RANGE 1000-1450", "C 4 inlines:  111 .. 133"])
    got = {(c.what, c.byte) for c in t.byte_claims}
    assert {("inline", 9), ("crossline", 21), ("x", 73), ("y", 77)} <= got
    assert {(r.what, r.lo, r.hi) for r in t.range_claims} >= {("inline", 1000, 1450), ("inline", 111, 133)}


def test_crs_hints_and_address_lines_ignored():
    t = parse(["C 1 DATUM ED-50 UTM ZONE 31N CM 3E", "C 2 AREA: NORTH SEA", "C 3 ACME, HOUSTON, TEXAS 77001 (713) 555-1234"])
    kinds = {(h.kind, h.value if h.kind != "utm_zone" else tuple(h.value)) for h in t.hints}
    assert ("datum", "ED50") in kinds and ("utm_zone", (31, "N")) in kinds and ("region", "NORTH SEA") in kinds
    assert ("region", "TEXAS") not in kinds


def test_crs_confirmed_from_text():
    x, y = np.array([620197.0, 620622.0]), np.array([6074232.0, 6074794.0])
    r = crs.resolve(x, y, "projected", parse(["C 1 ED50 UTM ZONE 31N NORTH SEA"]))
    assert r.status == "confirmed" and r.chosen.epsg == 23031


def test_crs_zone_inferred_from_region_but_datum_open():
    x, y = np.array([620197.0, 620622.0]), np.array([6074232.0, 6074794.0])
    r = crs.resolve(x, y, "projected", parse(["C 1 F3 block, Dutch North Sea"]))
    assert r.status == "ambiguous"
    assert {c.epsg for c in r.candidates[:3]} == {23031, 32631, 25831}  # zone 31N, datum undecided


def test_crs_contradiction_refused():
    x, y = np.array([620197.0, 620622.0]), np.array([6074232.0, 6074794.0])
    r = crs.resolve(x, y, "projected", parse(["C 1 WGS84 UTM ZONE 15N GULF OF MEXICO"]))
    assert r.chosen is None


def test_land_sea():
    assert crs.on_land(5.5, 52.0) and not crs.on_land(4.8, 54.8)


def test_region_box_across_antimeridian():
    """Regression: the Bering Sea spans the 180 degree meridian; a survey at 178E is inside it, not a conflict."""
    from segwise.crs import _in_box
    from segwise.textheader import REGIONS

    bering = REGIONS["BERING SEA"]
    assert _in_box(178.2, 58.2, bering) and _in_box(-170.0, 58.0, bering)
    assert not _in_box(150.0, 58.0, bering) and not _in_box(-150.0, 58.0, bering)
    assert _in_box(5.0, 56.0, (1.0, 51.0, 9.0, 62.0)) and not _in_box(20.0, 56.0, (1.0, 51.0, 9.0, 62.0))
