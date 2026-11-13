import numpy as np
import pytest

from segwise.formats import FORMATS, decode_samples, float_to_ibm, ibm_to_float
from conftest import SEGYIO, needs_segyio_data


def test_ibm_known_values():
    # 0x42640000 = 100.0, 0xC2640000 = -100.0, 0x41100000 = 1.0, 0 = 0
    w = np.array([0x42640000, 0xC2640000, 0x41100000, 0], dtype=np.uint32)
    assert ibm_to_float(w).tolist() == [100.0, -100.0, 1.0, 0.0]


def test_ibm_roundtrip():
    rng = np.random.default_rng(0)
    v = (rng.normal(size=10000) * 10.0 ** rng.integers(-20, 20, 10000)).astype(np.float32)
    back = ibm_to_float(float_to_ibm(v))
    assert np.allclose(back, v, rtol=1e-6)


@needs_segyio_data
def test_ibm_matches_segyio():
    import segyio

    from segwise.reader import TraceReader, probe

    path = f"{SEGYIO}/small.sgy"
    with segyio.open(path, ignore_geometry=True) as f:
        ref = segyio.tools.collect(f.trace[:])
    lay = probe(path)
    ours = TraceReader(lay).samples(np.arange(lay.n_traces))
    assert np.array_equal(ours, ref)


@needs_segyio_data
@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16])
@pytest.mark.parametrize("order", ["msb", "lsb"])
def test_all_formats_decode(code, order):
    """Every sample format, both byte orders, checked against an independent reference."""
    import warnings

    import segyio

    from segwise.amplitude import verify_format
    from segwise.reader import TraceReader, probe

    path = f"{SEGYIO}/multiformats/Format{code}{order}.sgy"
    lay = probe(path)
    assert lay.format_code == code and lay.n_traces == 414
    R = TraceReader(lay)
    idx = np.arange(414)
    if code == 4:
        # obsolete code; this file actually holds IEEE floats (segyio falls back to IBM and mis-decodes it).
        v = verify_format(R, lay)
        assert v.changed and v.effective == 5
        ref = TraceReader(probe(f"{SEGYIO}/multiformats/Format5{order}.sgy")).samples(idx)
        assert np.array_equal(R.samples(idx, 5), ref)
        return
    ours = R.samples(idx).astype(float)
    if code in (7, 15):
        # 3-byte integers: segyio cannot decode these; compare with the same data stored as 4-byte ints
        ints = TraceReader(probe(f"{SEGYIO}/multiformats/Format2{order}.sgy")).samples(idx).astype(float)
        assert np.array_equal(ours, ints if code == 7 else np.mod(ints, 2**24))
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with segyio.open(path, ignore_geometry=True, endian="big" if order == "msb" else "little") as f:
            ref = segyio.tools.collect(f.trace[:]).astype(float)
    # analysis decodes to float32, so very large unsigned integers agree to ~7 significant digits
    assert np.allclose(ours, ref, rtol=1e-6, atol=0)
