import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
DATA = os.environ.get("SEGWISE_TESTDATA", os.path.join(HERE, "data"))
SEGYIO = os.path.join(DATA, "segyio")
NAMSS = os.path.join(DATA, "namss")

needs_segyio_data = pytest.mark.skipif(not os.path.isdir(SEGYIO), reason="run scripts/fetch_test_data.sh")
needs_namss = pytest.mark.skipif(not os.path.isdir(NAMSS), reason="run scripts/fetch_test_data.sh --with-namss")


@pytest.fixture(scope="session")
def torture_dir(tmp_path_factory):
    from synth import cases, write

    d = tmp_path_factory.mktemp("torture")
    truths = {}
    for c in cases():
        p = str(d / f"{c.name}.sgy")
        truths[c.name] = (c, p, write(c, p))
    return truths
