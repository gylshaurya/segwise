# How segwise works

This explains each piece of detective work, what evidence it uses, and how it can fail. The guiding rule: **every answer must be tested against the data; text and headers are only claims.**

## 1. A SEG-Y file in one minute

```
 3200 bytes   textual header      40 lines x 80 characters, EBCDIC or ASCII, free text
  400 bytes   binary header       file-wide numbers: sample interval, samples per trace, sample format, revision...
 (n x 3200)   extended text       optional (rev 1+)
 then, repeated for every trace:
  240 bytes   trace header        ~90 numeric fields: inline, crossline, CDP, coordinates, scalar...
  ns x 1-8    samples             the seismic amplitudes for this trace
```

Byte positions are quoted 1-based, as in the standard (e.g. "inline at 189" means bytes 189-192 of each trace header).

## 2. Physical layout (`reader.py`)

**Byte order.** The binary header is parsed both big- and little-endian. Each reading is scored for sanity: a valid format code, a plausible sample count and interval, and the rev 2 byte-order marker (0x01020304). The more sensible reading wins.

**Trace length.** Candidate sample counts come from the binary header, the rev 2 extended count, the first trace header and the "original" count. Each is combined with each sample width (1, 2, 3, 4, 8 bytes) and each plausible count of extended text headers, including text blocks that are present but undeclared. A candidate scores highly when:

- the file size divides exactly into traces,
- the predicted trace headers carry sensible values, and
- **consecutive traces carry consecutive sequence numbers**.

The last rule matters because coincidences happen. In one test file the stale 1500-sample length divides the file exactly too, as 6 × the real 200-sample trace length.

Truncated files, files with no traces, variable-length traces (rev 2), Seismic Unix files (no file headers) and rev 2.1 extension headers are all handled. Files are memory-mapped read-only and the original is never modified.

## 3. What each header field holds (`geometry.py`)

A sample of trace headers is read in **contiguous blocks** spread through the file. Contiguity matters: "inline stays constant while crossline counts up" can only be seen in consecutive traces. Every 2-byte and 4-byte position is described: constant or varying, number of distinct values, how often it changes, its typical step, and which other positions hold identical values.

**3D grid.** Look for two integer fields (slow, fast) such that:

1. the pairs are unique for more than 97% of traces,
2. the fast key *resets* when the slow key changes. That is what distinguishes a grid from a 2D line numbered by shotpoint and CDP, where the second number just keeps counting,
3. two other fields fit **X = x0 + a·inline + b·crossline** (and likewise Y) with residuals below a fraction of a bin, giving a sensible, roughly rectangular bin.

Point 3 is the decisive proof. Wrong fields essentially cannot fit a plane to rounding precision. It also delivers the bin sizes, grid azimuth, handedness and corner coordinates. Traces with null (0,0) coordinates are excluded from the fit, then flagged and repaired.

Guards learned from real files: trace counters (bytes 1, 5) fit a plane in cropped grids, so known counter fields are never coordinates. Halves of large 4-byte numbers are not keys. A 4-byte reading straddling two standard 2-byte fields is ignored unless it really looks like a coordinate. When the same value is stored in several places (inline at 9 *and* 189), the conventional location is preferred and the copies are reported.

**Which key is the inline?** Data reveal the sort order, not the name. The slow key is assumed to be the inline unless it sits at byte 193 and the fast key at 189, or the textual header says otherwise.

**2D lines.** One key unique per trace and stepping regularly, with X/Y moving smoothly (near-constant spacing per CDP).

**Gathers.** A varying offset (byte 37), or ensemble keys (FFID 9, CDP 21) that repeat in runs with a channel number cycling inside them.

## 4. Sample format (`amplitude.py`)

Samples from ~200 traces are decoded with every format of the same byte width and scored for "seismic-likeness": finite values, a sensible dynamic range, neighbouring samples correlated, no wild jumps in magnitude.

That score alone **cannot** separate IBM from IEEE floats: reading one as the other is a smooth, order-preserving distortion, so the image still looks like seismic. That is exactly why the mistake survives in industry. The decisive test is structural. A normalised IBM float always has a non-zero leading hexadecimal digit in its fraction. Over thousands of samples, real IEEE data shows a zero digit 5-10% of the time and real IBM data 0%. (In testing: IEEE 8.5%, IBM 0.000% across over a million real 1986 samples.)

Amplitude QC then counts dead traces, NaN/Inf, clipping (flat tops at the maximum), spikes, and IBM values too large for IEEE (found in the real 1986 data).

## 5. Coordinates and scalar (`analyze.py`)

The coordinate scalar (bytes 71-72) is applied per trace. A positive value multiplies; a negative one divides. If the resulting bin or trace spacing isn't a standard survey size **and** the coordinate magnitudes are implausible, other powers of ten are tried. A scalar is only changed automatically when one gives a standard bin *and* plausible magnitudes; otherwise it is reported for checking. Swapped X/Y is detected from UTM-style magnitudes, or because only the swapped order lands inside the stated CRS.

## 6. CRS (`crs.py`, `textheader.py`)

The text parser extracts EPSG codes, datum names (ED50, NAD27, WGS 84, GDA94... ~30), spheroids, projections, UTM zones and central meridians, and place names (~70 regions with lon/lat boxes; contact addresses are ignored). These are turned into EPSG candidates using pyproj's offline EPSG database. Each candidate is scored by:

- whether the survey falls inside the CRS's official area of use,
- whether it falls inside the regions the text mentions,
- how directly the text states it.

If the zone isn't stated, the zones overlapping the mentioned regions are tried. For example, "Dutch North Sea" narrows UTM 30/31/32 to 31. Datums can't be distinguished from coordinates, so an unstated datum is always left as a decision for a person.

A world map drawn from bundled Natural Earth coastlines shows where the survey lands. No online tiles are used, because tile requests would reveal the survey's location to a third party.

## 7. Writing and verifying (`writer.py`)

The output is SEG-Y rev 1, big-endian, fixed length:

- inline → 189, crossline → 193, CDP X/Y → 181/185 (also 73/77 when those were empty), scalar → 71
- a corrected binary header, and the sample count and interval in every trace header
- a provenance stamp in blank lines of the textual header (the original text is kept)
- samples copied bit-for-bit, or converted to IEEE on request (values IEEE can't represent are set to 0 and reported)

It is written to a temporary file and renamed atomically. Verification then re-reads the output and checks:

- the structure,
- the trace count,
- the samples (compared byte-for-byte),
- the keys against the source,
- that coordinates sit on the analysed grid,
- and, if installed, that **segyio** (an independent reader) recovers the 3D geometry with default settings.

## 8. Navigation merge (`nav.py`)

UKOOA P1/90 and delimited tables are parsed (coordinate columns are recognised by their value ranges). Each nav line and integer column is tried against the SEG-Y's FFID, shotpoint and CDP fields. Because key numbering restarts on every line, the score combines overlap with similarity of line extent and a structural file-name match (`f2860001.sgy` ↔ `F286-1`). Gaps are linearly interpolated; points outside the navigated range are never extrapolated. Lat/long is projected to the line's UTM zone (or a CRS you choose), and an unstated nav datum is reported.
