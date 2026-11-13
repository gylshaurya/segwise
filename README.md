# Segwise

**Load any SEG-Y file right the first time.** Segwise works out where the geometry, coordinates and coordinate system of a SEG-Y file *really* are, proves every answer with evidence from the file itself, and writes a clean standard copy that loads with default settings in Petrel, OpendTect, Kingdom, DUG Insight, segyio or anything else.

Made for **seismic data managers** who receive deliveries of tens to thousands of files, and for **geophysicists and students** who just want a volume to load.

Everything runs on your computer. Files are read where they are and are never uploaded.

## Download

| | |
|---|---|
| **Windows** | Download **[Segwise.exe](https://github.com/gylshaurya/segwise/releases/latest/download/Segwise.exe)** and double-click it. No installation needed. |
| **Mac / Linux** | `pipx install "segwise[ui] @ git+https://github.com/gylshaurya/segwise"`, then run `segwise ui` |

On Windows, the first launch may show "Windows protected your PC" because the app is new and not yet code-signed. Click **More info**, then **Run anyway**. A small black window opens alongside your browser: keep it open while you work, and close it to quit.

## How to use it

1. **Check.** Click **Choose a file** or **Choose a folder** (a whole delivery) and pick it in the normal file window. Segwise checks it straight away.
2. **Read the coloured banner.** It always gives one of four answers and tells you the next step:
   - **Ready**: load it as it is, using the settings under *How to load it*.
   - **Fixable**: click **Write a fixed copy**.
   - **Needs a decision**: something can't be known from the file alone, almost always the coordinate system. Pick it from the short list in the banner (it is usually written in the delivery letter) and click **Use this system**. Segwise checks the file again with your choice and warns you if it puts the survey somewhere other than where the file says it should be. For a folder, set it once for every file with **Apply to all files**.
   - **Can't load**: the file is broken, and the banner says why.
3. **Write a fixed copy.** You get a standard SEG-Y file that includes your decisions, plus a report you can print or email, a load sheet (every decision and its evidence) and a map outline for QGIS or ArcGIS. The fixed file is checked after writing: amplitudes identical to the original, geometry readable with default settings. Your original file is never changed.

## Why this exists

SEG-Y is the 1975 file format that almost all seismic data is still exchanged in. Each trace carries a 240-byte header of numbers whose positions are *supposed* to be standard but in practice vary by vendor, decade and software. Before a file can be used, someone must work out:

- which bytes hold the **inline** and **crossline** numbers (the grid position of each trace),
- which bytes hold the **X/Y coordinates**, and by what **scalar** they must be divided,
- which **coordinate reference system** (datum + projection + zone) the coordinates are in,
- whether the declared **sample format** (IBM vs IEEE float, ...) is even true.

The usual tools show you the header bytes and leave the detective work to a person. The 40-line text note at the top of the file, where people write this information, is frequently stale or wrong. Getting it wrong means a survey loads in the wrong place, mirrored, or with distorted amplitudes, and nobody notices until much later.

segwise does the detective work and shows its reasoning.

## What it does

| | |
|---|---|
| **Finds the geometry from the data** | Tests every 2- and 4-byte header position. A 3D grid is proven when two fields form unique (inline, crossline) pairs **and** two other fields are an *exact* linear function of them (the X/Y coordinates). A wrong guess essentially cannot fit a plane to rounding precision. Also recognises 2D lines, pre-stack gathers and shot records. |
| **Checks the textual header against the data** | Byte locations, line ranges, sample rate, datum, UTM zone and region mentioned in the text are extracted as *claims* and tested. Contradictions are reported with the line they came from. |
| **Proves the sample format** | Detects IBM floats mislabelled as IEEE (or the reverse) with a normalisation test that is essentially conclusive. Genuine IBM floats never have a zero leading hex digit; IEEE data does about 5-10% of the time. |
| **Checks the coordinate scalar** | Flags scalars that make the bin size or coordinate magnitudes implausible, and proposes the right one. |
| **Resolves the CRS honestly** | Turns textual clues into EPSG codes, checks the survey lands inside the CRS's area of use and the region the text names, and draws it on an offline world map. When the data cannot decide (e.g. the datum), it says so instead of guessing. |
| **Writes a fixed, verified copy** | Big-endian SEG-Y rev 1, keys at the standard bytes (189/193/181/185), correct binary header. Amplitudes are copied bit-for-bit. The output is then re-read and checked, independently with segyio as well. The original is never modified. |
| **Merges navigation** | For legacy lines with no coordinates in the headers: reads UKOOA P1/90 or CSV/whitespace navigation, matches it to the traces by FFID/shotpoint/CDP *from the data*, and writes positioned files. |
| **Batch inventory** | Scans a whole delivery in parallel. Produces a CSV, GeoJSON footprints for GIS and an HTML index, and groups files that share a header layout so one confirmed load sheet can be applied to all of them. |
| **Load sheets** | A JSON record of every decision, its evidence and confidence. It travels with the data, is archived for audit, and can be re-applied to other files. |
| **Works how you work** | A local web app, a command line with script-friendly exit codes, single-file HTML reports (print to PDF), a Python API, and an MCP server for AI assistants. |

## Privacy

Everything runs on your machine. Files are read in place and never uploaded; the web app listens on `127.0.0.1` only. Maps are drawn from bundled coastlines, so no map-tile requests reveal where your survey is. Segwise sends nothing anywhere.

## Command line

Install with Python 3.10+ on Windows, macOS or Linux:

```bash
pipx install "segwise[ui] @ git+https://github.com/gylshaurya/segwise"
```

(`Segwise.exe` on Windows accepts the same commands, e.g. `Segwise.exe scan survey.sgy`.)

### Commands

```bash
segwise scan survey.sgy                 # verdict, how to load it, issues; writes report + load sheet
segwise scan /data/delivery/            # every SEG-Y under a folder
segwise fix survey.sgy -o survey_std.sgy --crs EPSG:23031
segwise inventory /data/delivery -o inventory/   # CSV, GeoJSON, HTML index, per-file reports
segwise fix line12.sgy --nav survey.p190 -o line12_nav.sgy
segwise headers survey.sgy --bytes 9,21,73,77,71 -n 50
segwise ui                              # local web app at http://127.0.0.1:8765
```

Every decision can be overridden: `--inline 9 --crossline 21 --x 73 --y 77 --scalar -100 --crs 23031 --format 1 --dt 4 --swap-xy`.
Apply a confirmed load sheet to a whole delivery: `segwise fix --loadsheet confirmed.loadsheet.json /data/delivery -o /data/fixed`.

### Exit codes (for ingestion pipelines)

| code | meaning |
|---|---|
| 0 | ready, or fixable automatically |
| 1 | needs a human decision (e.g. CRS ambiguous, no coordinates) |
| 2 | unusable, or verification failed |

### Python

```python
from segwise.analyze import analyze
from segwise.writer import write_fixed

report = analyze("survey.sgy", overrides={"epsg": 23031})
print(report.status, report.geometry["inline"], report.crs["chosen"])
log = write_fixed(report, "survey_std.sgy")
assert log.verification["passed"]
```

### AI assistants (MCP)

`segwise mcp` runs Segwise as a Model Context Protocol server (install with `segwise[mcp]`), so AI assistants that support MCP can scan and fix files.
Tools: `scan_segy`, `fix_segy`, `inventory_segy`, `header_values`, `merge_navigation`.

## Outputs

- `<file>.report.html`: single offline file with verdict, load settings, survey corners, maps, section preview, issues explained in plain language, evidence, annotated text header and header map. Prints to PDF.
- `<file>.loadsheet.json`: every decision with evidence; see [docs/LOADSHEET.md](docs/LOADSHEET.md).
- `<file>.footprint.geojson`: survey outline in WGS 84 for QGIS/ArcGIS.
- `<file>_std.sgy`: the fixed file (with `fix`).
- `inventory.csv`, `inventory.geojson`, `index.html`, `groups.json`: from `inventory`.

## How well does it work?

The test suite (`pytest`, 134 tests) includes:

- **27 synthetic "torture" files** with known answers: legacy byte locations (9/21/73/77), non-standard locations (221/225, 201/205), 2-byte keys, little-endian, IBM/IEEE mislabels, wrong scalar, swapped X/Y, crossline-sorted, irregular outline, null coordinates, truncated file, stale sample counts, a lying textual header, float coordinates, 2D lines, shot gathers. Each is analysed, fixed, re-read and compared to the truth.
- **segyio's public test corpus**, including every sample format in both byte orders, rev 2.1 extension headers and Seismic Unix files.
- **Real data**: a cropped F3 Netherlands cube (located correctly in the Dutch North Sea; the fixed copy loads in segyio with default settings, sample-for-sample identical) and the USGS NAMSS F-2-86-BS Bering Sea survey (39 real 1986 lines without header coordinates; navigation is matched to the correct line for 39/39).
- A 4.2 GB, 1,000,000-trace survey scans in about 18 s and is fixed and verified in about 22 s on a laptop.

Run `scripts/fetch_test_data.sh --with-namss` to download the public data, then `pytest`.

## Limits (honest)

- A **datum** can't be inferred from coordinates alone (datums differ by ~10-300 m); if it isn't stated, segwise asks.
- Pre-stack and shot-gather files are recognised and standardised, but get less geometry QC than stacked volumes.
- Very irregular or highly decimated surveys may need the key bytes set by hand (`--inline`/`--crossline`).
- SEG-Y rev 2 trace-header extensions are read but not carried into the rev 1 output.

## Documentation

- [How it works](docs/HOW_IT_WORKS.md): the detective methods, in detail
- [Load sheet format](docs/LOADSHEET.md)

## Feedback

Found a file Segwise gets wrong, or something confusing? Please [open an issue](https://github.com/gylshaurya/segwise/issues). Files it gets wrong are the most useful thing you can send; if you can't share the file, the load sheet (`.loadsheet.json`) usually tells the story.

## Licence

Apache-2.0. Bundled coastlines from Natural Earth (public domain).
