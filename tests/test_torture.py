"""Synthetic files with known answers, covering the layouts real deliveries contain."""
import numpy as np
import pytest

from segwise.analyze import analyze
from segwise.reader import TraceReader, probe
from segwise.writer import write_fixed
from synth import cases

KIND = {"3d": "3D post-stack", "2d": "2D post-stack", "shots": "shot gathers"}
NAMES = [c.name for c in cases()]


@pytest.mark.parametrize("name", NAMES)
def test_analysis_matches_truth(torture_dir, name):
    c, path, truth = torture_dir[name]
    r = analyze(path)
    G = r.geometry
    assert G["kind"] == KIND[c.kind]
    assert r.samples["format"] == c.fmt
    assert r.layout["ns"] == c.ns
    if c.kind == "3d":
        assert (G["inline"]["byte"], G["crossline"]["byte"]) == (c.il_byte[0], c.xl_byte[0])
        assert G["il_min"] == truth["il"].min() and G["xl_max"] == truth["xl"].max()
        assert G["inline_spacing"] == pytest.approx(c.bin_il, abs=0.05)
        assert G["crossline_spacing"] == pytest.approx(c.bin_xl, abs=0.05)
    if c.kind in ("3d", "2d"):
        assert {G["x"]["byte"], G["y"]["byte"]} == {c.x_byte[0], c.y_byte[0]}
        assert bool(G.get("swap_xy")) == c.swap_xy
        if c.x_byte[1] != "float32":
            assert G["scalar_effective"] == c.effective_scalar
    if c.epsg and any("EPSG" in t for t in c.text):
        assert r.crs["chosen"]["epsg"] == c.epsg and r.crs["status"] == "confirmed"


@pytest.mark.parametrize("name", NAMES)
def test_fix_and_verify(torture_dir, tmp_path, name):
    c, path, truth = torture_dir[name]
    r = analyze(path)
    log = write_fixed(r, str(tmp_path / "out.sgy"))
    assert log.verification["passed"], log.verification
    lay = probe(log.output)
    assert lay.endian == ">" and not lay.issues
    R = TraceReader(lay)
    idx = np.arange(lay.n_traces)
    if c.kind == "3d":
        assert np.array_equal(R.field(idx, 189, "int32"), truth["il"][: lay.n_traces])
        assert np.array_equal(R.field(idx, 193, "int32"), truth["xl"][: lay.n_traces])
    if c.kind in ("3d", "2d"):
        f = log.output_scalar if log.output_scalar > 0 else 1 / abs(log.output_scalar)
        X = R.field(idx, 181, "int32") * f
        Y = R.field(idx, 185, "int32") * f
        n = lay.n_traces
        # precision limits of the source: float32 (~0.5 m at 7 digits) or the integer scalar (e.g. -10 = 0.1 m)
        tol = 0.6 if c.x_byte[1] == "float32" else 0.75 * (1 / abs(c.effective_scalar) if c.effective_scalar < 0 else 1)
        assert np.median(np.hypot(X - truth["x"][:n], Y - truth["y"][:n])) < tol
    # the fixed file needs no fixing
    r2 = analyze(log.output, overrides={"epsg": c.epsg} if c.epsg and c.kind != "shots" else None)
    assert not [i for i in r2.issues if i.auto_fixed and i.code != "SCALAR_ZERO"]
    if c.kind == "3d":
        assert (r2.geometry["inline"]["byte"], r2.geometry["crossline"]["byte"]) == (189, 193)
        assert (r2.geometry["x"]["byte"], r2.geometry["y"]["byte"]) == (181, 185)


def test_wrong_scalar_detected(torture_dir):
    r = analyze(torture_dir["wrong_scalar"][1])
    assert "SCALAR_WRONG" in {i.code for i in r.issues}


def test_lying_text_header_flagged(torture_dir):
    r = analyze(torture_dir["lying_text"][1])
    wrong = [i for i in r.issues if i.code == "TEXT_CLAIM_WRONG"]
    assert len(wrong) >= 2  # claims inline 189 / crossline 193; data says 9 / 21


def test_format_mislabels(torture_dir):
    assert {i.code for i in analyze(torture_dir["ieee_declared_ibm"][1]).issues} >= {"FORMAT_WRONG"}
    assert {i.code for i in analyze(torture_dir["ibm_declared_ieee"][1]).issues} >= {"FORMAT_WRONG"}


def test_original_never_modified(torture_dir, tmp_path):
    import hashlib

    path = torture_dir["little_endian"][1]
    before = hashlib.sha256(open(path, "rb").read()).hexdigest()
    write_fixed(analyze(path), str(tmp_path / "o.sgy"))
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == before


def test_refuses_to_overwrite_source(torture_dir):
    from segwise.writer import WriteError

    path = torture_dir["standard"][1]
    with pytest.raises(WriteError):
        write_fixed(analyze(path), path, overwrite=True)


def test_overrides_win(torture_dir):
    path = torture_dir["legacy_9_21_73_77"][1]
    r = analyze(path, overrides={"epsg": 32631, "scalar": -100})
    assert r.crs["chosen"]["epsg"] == 32631 and r.geometry["scalar_effective"] == -100


# ---------------------------------------------------------------- regressions from the independent review
def _append(path, out, extra: bytes = b"", times: int = 1):
    data = open(path, "rb").read()
    head, traces = data[:3600], data[3600:]
    with open(out, "wb") as f:
        f.write(head + traces * times + extra)
    return out


def test_duplicated_cube_is_3d_with_duplicates_not_2d(torture_dir, tmp_path):
    p = _append(torture_dir["standard"][1], str(tmp_path / "dup.sgy"), times=2)
    r = analyze(p)
    assert r.geometry["kind"] == "3D post-stack" and r.geometry["inline"]["byte"] == 189
    assert "DUPLICATE_TRACES" in {i.code for i in r.issues} and r.status == "needs_review"


def test_blank_trailing_trace_does_not_stretch_grid(torture_dir, tmp_path):
    c, path, truth = torture_dir["standard"]
    p = _append(path, str(tmp_path / "blank.sgy"), extra=bytes(240 + c.ns * 4))
    r = analyze(p)
    G = r.geometry
    assert (G["il_min"], G["il_max"]) == (truth["il"].min(), truth["il"].max())
    assert "BLANK_KEYS" in {i.code for i in r.issues}
    log = write_fixed(r, str(tmp_path / "o.sgy"))
    assert log.verification["passed"] and not any("Recomputed" in a for a in log.actions)


def test_two_joined_files_are_refused(torture_dir, tmp_path):
    from segwise.writer import WriteError

    a = open(torture_dir["standard"][1], "rb").read()
    p = tmp_path / "joined.sgy"
    p.write_bytes(a + a)
    r = analyze(str(p))
    assert "STRUCTURE_BREAK" in {i.code for i in r.issues} and r.status == "needs_review"
    with pytest.raises(WriteError):
        write_fixed(r, str(tmp_path / "o.sgy"))


def test_text_trailer_excluded(torture_dir, tmp_path):
    trailer = ("C 1 END OF FILE TRAILER STANZA".ljust(3200)).encode("cp037")
    p = _append(torture_dir["standard"][1], str(tmp_path / "trl.sgy"), extra=trailer)
    r = analyze(p)
    assert "TRAILER" in {i.code for i in r.issues} and r.geometry["kind"] == "3D post-stack"


def test_single_inline_has_coordinates(tmp_path):
    from synth import TEXT_GOOD, Case, write

    p = str(tmp_path / "one.sgy")
    write(Case("one", n_il=1, n_xl=200, text=TEXT_GOOD), p)
    r = analyze(p)
    assert r.geometry["kind"] == "3D post-stack" and r.geometry["x"]["byte"] == 181
    assert r.geometry["trace_spacing"] == pytest.approx(12.5, abs=0.01)


def test_large_float_coordinates_not_read_as_ints(tmp_path):
    from synth import Case, write

    p = str(tmp_path / "ft.sgy")
    truth = write(Case("ft", origin=(2_200_000.0, 13_500_000.0), x_byte=(201, "float32"), y_byte=(205, "float32"),
                       bin_il=82.5, bin_xl=82.5, epsg=None, text=["STATE PLANE FEET"]), p)
    r = analyze(p)
    assert (r.geometry["x"]["type"], r.geometry["y"]["type"]) == ("float32", "float32")
    log = write_fixed(r, str(tmp_path / "o.sgy"))
    lay = probe(log.output)
    R = TraceReader(lay)
    f = log.output_scalar if log.output_scalar > 0 else 1 / abs(log.output_scalar)
    X = R.field(np.arange(lay.n_traces), 181, "int32") * f
    assert np.max(np.abs(X - truth["x"])) < 2.0  # float32 precision at 2.2 million


def test_varying_scalar_handled(torture_dir, tmp_path):
    c, path, truth = torture_dir["standard"]
    data = bytearray(open(path, "rb").read())
    tb = 240 + c.ns * 4
    n = (len(data) - 3600) // tb
    for i in range(0, n, 2):  # every other trace: scalar 1 and whole-metre coordinates
        o = 3600 + i * tb
        data[o + 70 : o + 72] = np.array([1], ">i2").tobytes()
        data[o + 180 : o + 184] = np.array([round(truth["x"][i])], ">i4").tobytes()
        data[o + 184 : o + 188] = np.array([round(truth["y"][i])], ">i4").tobytes()
    p = tmp_path / "sv.sgy"
    p.write_bytes(bytes(data))
    r = analyze(str(p))
    assert r.geometry["x"]["byte"] == 181 and "SCALAR_VARIES" in {i.code for i in r.issues}
    assert r.geometry["inline_spacing"] == pytest.approx(c.bin_il, abs=0.2)
