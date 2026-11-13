import pytest

from segwise.reader import ProbeError, probe
from conftest import SEGYIO, needs_segyio_data

CASES = {
    # file: (endian, format, ns, n_traces, issue codes that must be present)
    "small.sgy": (">", 1, 50, 25, set()),
    "small-lsb.sgy": ("<", 1, 50, 25, {"LITTLE_ENDIAN"}),
    "f3.sgy": (">", 3, 75, 414, {"NS_MISMATCH_TRACE"}),
    "f3-lsb.sgy": ("<", 3, 75, 414, {"LITTLE_ENDIAN"}),
    "broken.sgy": (">", 1, 4, 3, {"TRUNCATED"}),
    "multi-text.sgy": (">", 1, 1, 1, set()),
    "stanzas-unknown-count.sgy": (">", 1, 4, 6, set()),
    "mapping-empty.sgy": (">", 1, 4, 6, {"EXT_TEXT_COUNT_WRONG"}),
    "trace-header-extensions.sgy": (">", 1, 4, 2, set()),
    "long.sgy": (">", 1, 60000, 3, set()),
}


@needs_segyio_data
@pytest.mark.parametrize("name", sorted(CASES))
def test_probe_corpus(name):
    e, fmt, ns, n, codes = CASES[name]
    lay = probe(f"{SEGYIO}/{name}")
    assert (lay.endian, lay.format_code, lay.ns, lay.n_traces) == (e, fmt, ns, n)
    assert codes <= {i.code for i in lay.issues}


@needs_segyio_data
@pytest.mark.parametrize("name,endian", [("small.su", ">"), ("small-lsb.su", "<")])
def test_seismic_unix(name, endian):
    lay = probe(f"{SEGYIO}/{name}")
    assert lay.is_su and lay.endian == endian and lay.n_traces == 25 and lay.ns == 50


@needs_segyio_data
def test_headers_only_file():
    lay = probe(f"{SEGYIO}/text.sgy")
    assert lay.n_traces == 0 and "NO_TRACES" in {i.code for i in lay.issues}


def test_not_segy(tmp_path):
    p = tmp_path / "x.sgy"
    p.write_bytes(b"hello")
    with pytest.raises(ProbeError):
        probe(str(p))
