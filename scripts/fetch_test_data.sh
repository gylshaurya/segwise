#!/usr/bin/env sh
# Download the public real-world test data used by the test suite (not redistributed in this repo).
#   tests/data/segyio/      segyio's test corpus, incl. a cropped F3 Netherlands cube (LGPL-3.0, Equinor)
#   tests/data/namss/       USGS NAMSS survey F-2-86-BS, Bering Sea 1986: 39 lines + navigation (public domain, ~1 GB)
set -e
cd "$(dirname "$0")/../tests"
mkdir -p data && cd data
if [ ! -d segyio ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/equinor/segyio.git segyio-repo
  (cd segyio-repo && git sparse-checkout set test-data)
  mv segyio-repo/test-data segyio && rm -rf segyio-repo
fi
if [ "$1" = "--with-namss" ] && [ ! -d namss ]; then
  curl -L -o namss.zip https://walrus.wr.usgs.gov/namss/data/1986/namss.F-2-86-BS.mcs.airgun.zip
  mkdir namss && (cd namss && unzip -q ../namss.zip) && rm namss.zip
fi
echo "Test data ready in $(pwd)"
