# Load sheet format (`segwise.loadsheet/1`)

A load sheet is the portable record of how to read one SEG-Y file and why. It is written next to every scanned or fixed file, embedded in every HTML report (download link), and accepted back with `--loadsheet` to apply the same decisions to other files.

```jsonc
{
  "schema": "segwise.loadsheet/1",
  "generated_by": "segwise 0.1.0",
  "generated_at": "2026-11-13T00:36:12+00:00",
  "source": { "path": "...", "file_name": "f3.sgy", "size_bytes": 165060, "fingerprint": "9f1c..." },
  "status": "fixable",                    // ready | fixable | needs_review | unusable
  "status_reason": "...",
  "summary": "3D post-stack, IL 111-133 x XL 875-892, ...",
  "layout": { "endian": "big", "ns": 75, "dt_us": 4000, "n_traces": 414, "sample_format": 3, "declared_format": 3, ... },
  "geometry": {
    "kind": "3D post-stack",
    "inline":    { "byte": 189, "type": "int32" },
    "crossline": { "byte": 193, "type": "int32" },
    "x": { "byte": 181, "type": "int32" }, "y": { "byte": 185, "type": "int32" },
    "scalar_header": -10, "scalar_effective": -10, "swap_xy": false,
    "il_min": 111, "il_max": 133, "il_inc": 1, "xl_min": 875, "xl_max": 892, "xl_inc": 1,
    "inline_spacing": 25.0, "crossline_spacing": 25.0, "inline_azimuth": 88.4,
    "corners": [ { "inline": 111, "crossline": 875, "x": 620197.23, "y": 6074232.87 }, ... ],
    "centre_lonlat": [4.87165, 54.80144]
  },
  "crs": { "status": "confirmed", "chosen": { "epsg": 23031, "name": "ED50 / UTM zone 31N", "reasons": [...] }, "candidates": [...] },
  "decisions": [ { "key": "inline", "display": "bytes 189-192 (int32), 111-133", "confidence": "high",
                   "source": "data", "evidence": ["..."] }, ... ],
  "issues": [ { "code": "NS_MISMATCH_TRACE", "severity": "warning", "title": "...", "detail": "...",
                "explain": "...", "fix": "...", "auto_fixed": true }, ... ],
  "load_original_file": { "Inline number": "189-192", "Coordinate scalar": "bytes 71-72 (apply from header; value -10)", ... },
  "overrides": { "inline": {...}, "crossline": {...}, "x": {...}, "y": {...}, "epsg": 23031 },
  "fixed_file": {                         // only after `fix`
    "path": ".../f3_std.sgy", "actions": [...], "warnings": [...], "output_scalar": -10, "output_format": 3,
    "verification": { "passed": true, "checks": [ { "name": "...", "ok": true, "detail": "..." } ] },
    "load": { "Inline number": "189-192", ... }
  }
}
```

**Re-using a sheet.** Only the `overrides` block is applied. It contains exactly the decisions that differ from what segwise would infer on its own, plus the CRS. Each file is still fully re-analysed, so a file in the batch that does not match the confirmed layout shows up with contradictions instead of being silently mis-loaded.

**Issue codes** are stable identifiers for automation: e.g. `TRUNCATED`, `FORMAT_WRONG`, `SCALAR_WRONG`, `XY_SWAPPED`, `TEXT_CLAIM_WRONG`, `CRS_AMBIGUOUS`, `NO_COORDINATES`, `DUPLICATE_TRACES`, `IBM_OVERFLOW`.
