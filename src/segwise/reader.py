"""Low-level, read-only access to a SEG-Y (or Seismic Unix) file.

`probe()` works out the physical layout of the file -- byte order, sample
format, trace length, number of traces, extended headers -- by testing the
binary header's claims against the file itself rather than trusting them.
The original file is opened read-only through a memory map and never modified.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from .fields import BIN
from .formats import FORMATS, SAME_WIDTH, decode_samples
from .model import Issue

TEXT_SIZE = 3200
BIN_SIZE = 400
FILE_HEADER = TEXT_SIZE + BIN_SIZE
TH = 240  # trace header size


# ----------------------------------------------------------------------------- text
_ASCII_PRINTABLE = set(range(0x20, 0x7F)) | {0x0A, 0x0D, 0x09}


def _ebcdic_printable_fraction(raw: bytes) -> float:
    txt = raw.decode("cp037", errors="replace")
    good = sum(1 for ch in txt if ch.isprintable() and ord(ch) < 0x250 or ch in "\n\r\t")
    letters = sum(1 for ch in txt if ch.isalnum())
    return 0.7 * good / max(len(txt), 1) + 0.3 * min(1.0, letters / max(len(txt) * 0.15, 1))


def _ascii_printable_fraction(raw: bytes) -> float:
    good = sum(1 for b in raw if b in _ASCII_PRINTABLE)
    letters = sum(1 for b in raw if (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A) or (0x30 <= b <= 0x39))
    return 0.7 * good / max(len(raw), 1) + 0.3 * min(1.0, letters / max(len(raw) * 0.15, 1))


def decode_text_header(raw: bytes) -> tuple[str, str, list[str]]:
    """Decode a 3200-byte textual header. Returns (encoding, text, 40 lines)."""
    if not raw.strip(b"\x00 \x40"):
        return "empty", "", [""] * 40
    e = _ebcdic_printable_fraction(raw)
    a = _ascii_printable_fraction(raw)
    if a >= e:
        enc = "ascii"
        txt = raw.decode("latin-1")
    else:
        enc = "ebcdic"
        txt = raw.decode("cp037", errors="replace")
    clean = "".join(ch if (ch.isprintable() or ch == "\n") else " " for ch in txt.replace("\r", ""))
    if clean.count("\n") >= 10:  # newline-delimited header rather than 80-column cards
        lines = [ln.rstrip() for ln in clean.split("\n")]
    else:
        clean = clean.replace("\n", " ")
        lines = [clean[i : i + 80].rstrip() for i in range(0, len(clean), 80)]
    lines = (lines + [""] * 40)[:40] if len(lines) <= 40 else lines
    quality = max(a, e)
    if quality < 0.55:
        enc = "binary/unreadable"
    return enc, "\n".join(lines), lines


# ----------------------------------------------------------------------------- binary header
def parse_binary_header(raw: bytes, endian: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, (byte, size) in BIN.items():
        off = byte - 3201
        chunk = raw[off : off + size]
        if size == 1:
            out[name] = chunk[0]
        else:
            out[name] = int(np.frombuffer(chunk, dtype=np.dtype(f"{endian}i{size}"))[0])
    # samples per trace and interval are unsigned in rev 2
    for k in ("samples_per_trace", "sample_interval", "orig_samples_per_trace"):
        if out[k] < 0:
            out[k] += 65536
    return out


def _endian_score(b: dict[str, int]) -> float:
    s = 0.0
    if b["format_code"] in FORMATS:
        s += 4
    if 0 < b["samples_per_trace"] <= 50000:
        s += 1.5
    if 0 < b["sample_interval"] <= 100000:
        s += 1
    if b["byte_order_marker"] == 0x01020304:
        s += 10
    if 0 <= b["n_ext_text_headers"] <= 100:
        s += 0.5
    if 0 <= b["sorting_code"] <= 9:
        s += 0.3
    if b["measurement_system"] in (0, 1, 2):
        s += 0.3
    return s


# ----------------------------------------------------------------------------- layout
@dataclass
class Layout:
    path: str
    size: int
    is_su: bool
    endian: str  # '>' big or '<' little
    text_encoding: str
    text: str
    text_lines: list[str]
    binary: dict[str, int]
    n_ext_text: int
    ext_text: list[str]
    data_offset: int
    n_extra_trace_headers: int
    format_code: int  # declared (or inferred if the declared one is invalid)
    ns: int
    dt_us: float
    trace_bytes: int  # fixed length; 0 when variable
    n_traces: int
    remainder: int  # bytes left over after the last complete trace
    trace_offsets: np.ndarray | None = None  # only for variable-length files
    issues: list[Issue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def bps(self) -> int:
        return FORMATS[self.format_code][0]

    @property
    def header_bytes(self) -> int:
        return TH * (1 + self.n_extra_trace_headers)


class ProbeError(Exception):
    pass


def _mm(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode="r")


def _i16(mm, off: int, endian: str) -> int:
    return int(np.frombuffer(bytes(mm[off : off + 2]), dtype=f"{endian}i2")[0])


def _i32(mm, off: int, endian: str) -> int:
    return int(np.frombuffer(bytes(mm[off : off + 4]), dtype=f"{endian}i4")[0])


def _header_coherence(mm, size: int, data_offset: int, trace_bytes: int, n: int, ns: int, endian: str) -> float:
    """How much do the trace headers at the predicted positions look like trace headers?"""
    if n < 1:
        return 0.0
    picks = sorted({0, 1, 2, n // 2, n - 1} & set(range(n)))
    score = 0.0
    seq = []
    for i in picks:
        off = data_offset + i * trace_bytes
        if off + TH > size:
            continue
        tns = _i16(mm, off + 114, endian) & 0xFFFF
        tdt = _i16(mm, off + 116, endian) & 0xFFFF
        if tns == ns:
            score += 0.6
        elif tns == 0:
            score += 0.3
        if 0 < tdt <= 100000:
            score += 0.3
        code = _i16(mm, off + 28, endian)
        if -1 <= code <= 50:
            score += 0.3
        seq.append((_i32(mm, off, endian), _i32(mm, off + 4, endian)))
    score = score / max(len(picks), 1)
    if len(seq) >= 3:
        # consecutive traces carry consecutive sequence numbers (bytes 1-4 or 5-8): strong proof of the trace length
        for k in (0, 1):
            if seq[1][k] == seq[0][k] + 1 and seq[2][k] == seq[1][k] + 1:
                score += 1.5
                break
    return score


def probe(path: str) -> Layout:
    size = os.path.getsize(path)
    if size < TH + 4:
        raise ProbeError(f"File is only {size} bytes, too small to be SEG-Y.")
    mm = _mm(path)
    issues: list[Issue] = []
    notes: list[str] = []

    head = bytes(mm[: min(size, FILE_HEADER)])
    enc, text, lines = decode_text_header(head[:TEXT_SIZE]) if size >= TEXT_SIZE else ("none", "", [""] * 40)
    bins = {e: parse_binary_header(head[TEXT_SIZE:FILE_HEADER], e) for e in (">", "<")} if size >= FILE_HEADER else {}
    scores = {e: _endian_score(b) for e, b in bins.items()}

    # ---- Seismic Unix (no file headers) ------------------------------------------------
    if not bins or max(scores.values()) < 4:
        su = _probe_su(mm, size, path)
        if su is not None:
            return su
        if not bins:
            raise ProbeError("File is smaller than the 3600-byte SEG-Y file header and is not Seismic Unix.")

    endian = ">" if scores[">"] >= scores["<"] else "<"
    b = bins[endian]
    if endian == "<":
        if b["byte_order_marker"] == 0x01020304:
            notes.append("Little-endian byte order declared by the rev 2 byte-order marker.")
        else:
            issues.append(Issue(
                "LITTLE_ENDIAN", "warning", "File is little-endian (byte-swapped)",
                "The binary header only makes sense when read little-endian, and the rev 2 byte-order marker is absent. "
                "Standard SEG-Y (rev 0/1) is big-endian.",
                fix="The fixed file is written big-endian, which every loader expects.",
                explain="Computers store multi-byte numbers in one of two orders. This file was written in the 'wrong' order "
                "for SEG-Y, so tools that assume the standard order read nonsense numbers.",
            ))

    if enc == "ascii":
        notes.append("Textual header is ASCII rather than EBCDIC.")
    elif enc in ("binary/unreadable", "empty"):
        issues.append(Issue(
            "TEXT_HEADER_UNREADABLE", "warning", "Textual header is empty or unreadable",
            f"The 3200-byte textual header decodes as neither EBCDIC nor ASCII ({enc}).",
            fix="No survey information (CRS, byte locations) can be taken from it; everything is inferred from the data.",
            explain="The 'cover note' at the start of the file, where people normally write the coordinate system and "
            "header layout, is blank or garbled.",
        ))

    # ---- extended textual headers --------------------------------------------------------
    rev = b["rev_major"]
    n_ext = b["n_ext_text_headers"]
    ext_text: list[str] = []
    if n_ext == -1:
        n_ext = _count_ext_text_variable(mm, size)
        notes.append(f"Variable number of extended textual headers; found {n_ext} ending with ((SEG: EndText)).")
    elif n_ext < 0 or n_ext > 200 or FILE_HEADER + n_ext * TEXT_SIZE > size:
        issues.append(Issue(
            "BAD_EXT_TEXT_COUNT", "warning", "Implausible extended textual header count",
            f"Binary header bytes 3505-3506 claim {n_ext} extended textual headers; treated as 0.",
            explain="A counter in the header says extra text blocks follow, but the number is impossible.",
        ))
        n_ext = 0
    elif n_ext > 0:
        ok = []
        for i in range(n_ext):
            blk = bytes(mm[FILE_HEADER + i * TEXT_SIZE : FILE_HEADER + (i + 1) * TEXT_SIZE])
            e2, t2, _ = decode_text_header(blk)
            ok.append(e2 in ("ascii", "ebcdic"))
            ext_text.append(t2)
        if not all(ok):
            issues.append(Issue(
                "EXT_TEXT_NOT_TEXT", "warning", "Extended textual headers do not look like text",
                f"{n_ext} extended textual headers are declared but at least one block is not readable text; "
                "the count is probably a stale value from a different revision.",
                explain="The header claims extra text pages follow, but they look like binary data.",
            ))
            # decide below by trying both
    data_offsets = [FILE_HEADER + n_ext * TEXT_SIZE]
    if n_ext:
        data_offsets.append(FILE_HEADER)
    # text blocks that are present but not declared (or declared wrongly)
    k = 0
    while k < 20 and FILE_HEADER + (k + 1) * TEXT_SIZE <= size:
        blk = bytes(mm[FILE_HEADER + k * TEXT_SIZE : FILE_HEADER + (k + 1) * TEXT_SIZE])
        if decode_text_header(blk)[0] not in ("ascii", "ebcdic"):
            break
        k += 1
    for j in range(1, k + 1):
        if FILE_HEADER + j * TEXT_SIZE not in data_offsets:
            data_offsets.append(FILE_HEADER + j * TEXT_SIZE)

    n_extra = 0
    if rev >= 2:
        raw_extra = head[3506:3510]
        for v in (int(np.frombuffer(raw_extra, f"{endian}i4")[0]), int(np.frombuffer(raw_extra[:2], f"{endian}i2")[0])):
            if 0 < v < 100:
                n_extra = v
                notes.append(f"SEG-Y rev 2 file with {v} additional 240-byte trace header(s) per trace.")
                break

    # ---- candidate sample counts and formats ----------------------------------------------
    first_ns = {e: (_i16(mm, data_offsets[0] + 114, e) & 0xFFFF) for e in (">", "<")} if size >= data_offsets[0] + TH else {">": 0, "<": 0}
    first_dt = (_i16(mm, data_offsets[0] + 116, endian) & 0xFFFF) if size >= data_offsets[0] + TH else 0
    ns_cands: list[tuple[int, str]] = []
    if b["samples_per_trace"] > 0:
        ns_cands.append((b["samples_per_trace"], "binary header bytes 3221-3222"))
    if rev >= 2 and b["ext_samples_per_trace"] > 0:
        ns_cands.append((b["ext_samples_per_trace"], "binary header extended sample count (rev 2)"))
    for doff_c in data_offsets:
        if size >= doff_c + TH:
            v = _i16(mm, doff_c + 114, endian) & 0xFFFF
            if v > 0 and all(v != c[0] for c in ns_cands):
                ns_cands.append((v, "first trace header bytes 115-116"))
    if b["orig_samples_per_trace"] > 0:
        ns_cands.append((b["orig_samples_per_trace"], "binary header 'original' sample count"))

    declared_fmt = b["format_code"]
    fmt_cands: list[int] = [declared_fmt] if declared_fmt in FORMATS else []
    for w in (4, 2, 1, 8, 3):
        for c in SAME_WIDTH[w][:1]:  # one representative per width; refined later from the samples
            if c not in fmt_cands:
                fmt_cands.append(c)

    best = None
    for doff in data_offsets:
        for ns, ns_src in ns_cands:
            for fc in fmt_cands:
                tb = TH * (1 + n_extra) + ns * FORMATS[fc][0]
                payload = size - doff
                if payload < tb:
                    continue
                n, rem = divmod(payload, tb)
                coh = _header_coherence(mm, size, doff, tb, n, ns, endian)
                score = (6.0 if rem == 0 else (2.0 if rem < tb else 0.0)) + 3 * coh
                if fc == declared_fmt:
                    score += 1.5
                if ns_src.startswith("binary header bytes"):
                    score += 0.5
                if doff == data_offsets[0]:
                    score += 0.5
                cand = (score, doff, ns, ns_src, fc, tb, n, rem)
                if best is None or cand[0] > best[0]:
                    best = cand

    variable = None
    if best is None or best[0] < 6.5 or b["fixed_length_flag"] == 0 and rev >= 2:
        fc_v = declared_fmt if declared_fmt in FORMATS else 5
        variable = _walk_variable(mm, size, data_offsets[0], FORMATS[fc_v][0], endian, n_extra)
        if variable is not None and (best is None or best[0] < 6.5 or best[7] != 0):
            offs, ns_list = variable
            ns_mode = int(np.bincount(ns_list).argmax())
            if len(set(ns_list.tolist())) > 1:
                issues.append(Issue(
                    "VARIABLE_TRACE_LENGTH", "warning", "Traces have different lengths",
                    f"Trace lengths vary between {ns_list.min()} and {ns_list.max()} samples; "
                    f"most common is {ns_mode}.",
                    fix="The fixed file pads or trims every trace to the most common length so that it is a regular "
                    "fixed-length SEG-Y file.",
                    explain="Most software expects every trace in a file to have the same number of samples.",
                ))
            dt = b["sample_interval"] or first_dt
            lay = Layout(path, size, False, endian, enc, text, lines, b, n_ext, ext_text, data_offsets[0], n_extra,
                         fc_v, ns_mode, float(dt), 0, len(offs), 0, offs, issues, notes)
            lay.notes.append("Variable-length traces: an index of trace positions was built by walking the file.")
            return lay

    if best is None:
        if size - data_offsets[0] < TH:
            lay = Layout(path, size, False, endian, enc, text, lines, b, n_ext, ext_text, data_offsets[0], n_extra,
                         declared_fmt if declared_fmt in FORMATS else 1, b["samples_per_trace"], float(b["sample_interval"]),
                         0, 0, size - data_offsets[0], None, issues, notes)
            issues.append(Issue("NO_TRACES", "error", "File contains no traces",
                                "The file holds only file headers; there is no trace data after them.",
                                explain="This is an empty survey file: the cover pages are there but no seismic data."))
            return lay
        raise ProbeError("Could not find any trace length consistent with this file. It may be truncated or not SEG-Y.")

    score, doff, ns, ns_src, fc, tb, n, rem = best
    first_ns = {e: (_i16(mm, doff + 114, e) & 0xFFFF) for e in (">", "<")}
    first_dt = _i16(mm, doff + 116, endian) & 0xFFFF
    if rem:
        # a remainder can also mean mixed trace lengths (rev 1 files written that way) - try walking the file
        fc_w = fc
        walked = _walk_variable(mm, size, doff, FORMATS[fc_w][0], endian, n_extra)
        if walked is not None and len(set(walked[1].tolist())) > 1:
            offs, ns_list = walked
            ns_mode = int(np.bincount(ns_list).argmax())
            issues.append(Issue(
                "VARIABLE_TRACE_LENGTH", "warning", "Traces have different lengths",
                f"Trace lengths vary between {ns_list.min()} and {ns_list.max()} samples; most common is {ns_mode}.",
                fix="The fixed file pads or trims every trace to the most common length.",
                explain="Most software expects every trace in a file to have the same number of samples."))
            dt = b["sample_interval"] or first_dt
            return Layout(path, size, False, endian, enc, text, lines, b, n_ext, ext_text, doff, n_extra, fc, ns_mode,
                          float(dt), 0, len(offs), 0, offs, issues, notes)
    brk = _structure_break(mm, size, doff, tb, n, endian)
    if brk is not None:
        k, kind = brk
        tail = size - (doff + k * tb)
        if kind == "trailer":
            issues.append(Issue(
                "TRAILER", "info", "Text block after the last trace",
                f"{tail} bytes of text follow trace {k} (a SEG-Y rev 2 trailer stanza or appended notes).",
                fix="Excluded from the traces; not copied to the fixed file.",
                explain="Some files carry extra notes after the data. They are not seismic traces.", auto_fixed=True))
        else:
            where = "A second SEG-Y file header starts there; two files were probably joined." if kind == "second_file" \
                else "The trace layout changes there (mixed trace lengths or corruption)."
            issues.append(Issue(
                "STRUCTURE_BREAK", "error", "The file's structure breaks part-way through",
                f"Traces 1-{k} are consistent, but the bytes after trace {k} (offset {doff + k * tb:,}) no longer line up as "
                f"traces of {tb} bytes. {where}",
                fix="Only the first consistent traces are analysed. Split the file at that offset (or ask for a clean "
                "copy) before loading; Segwise will not write a fixed copy of a broken file.",
                explain="Part-way through, the file stops being the same kind of data, like two documents stapled "
                "together. Loading it as one file would turn everything after the join into garbage."))
        n, rem = k, 0
    if doff != data_offsets[0]:
        real = (doff - FILE_HEADER) // TEXT_SIZE
        issues.append(Issue(
            "EXT_TEXT_COUNT_WRONG", "warning", "Extended textual header count is wrong",
            f"The binary header says {b['n_ext_text_headers']} extended textual headers follow, but the traces only line up "
            f"when there are {real}.",
            fix=f"Used {real}; the fixed file declares the correct count.",
            explain="A counter claims extra text pages exist; they don't, and believing it would shift every trace.",
            auto_fixed=True,
        ))
        n_ext = real
        ext_text = [decode_text_header(bytes(mm[FILE_HEADER + i * TEXT_SIZE : FILE_HEADER + (i + 1) * TEXT_SIZE]))[1]
                    for i in range(real)]

    if declared_fmt not in FORMATS:
        issues.append(Issue(
            "BAD_FORMAT_CODE", "error", "Invalid sample format code",
            f"Binary header bytes 3225-3226 contain {declared_fmt}, which is not a SEG-Y format code. "
            f"The file size is consistent with {FORMATS[fc][0]}-byte samples.",
            fix="The real format is determined from the sample values themselves (see 'Sample format').",
            explain="The header doesn't say how the numbers in each trace are encoded, so a loader can't decode them.",
            auto_fixed=True,
        ))
    elif fc != declared_fmt:
        issues.append(Issue(
            "FORMAT_WIDTH_MISMATCH", "error", "Sample format code contradicts the file size",
            f"The header declares format {declared_fmt} ({FORMATS[declared_fmt][0]} bytes/sample) but the traces only line "
            f"up with {FORMATS[fc][0]} bytes/sample.",
            fix="The format is determined from the sample values.",
            explain="The header says each sample takes a certain number of bytes; the file's size proves it doesn't.",
            auto_fixed=True,
        ))

    if b["samples_per_trace"] != ns:
        issues.append(Issue(
            "NS_MISMATCH_BINARY", "error" if b["samples_per_trace"] else "warning",
            "Samples-per-trace in the binary header is wrong" if b["samples_per_trace"] else "Samples-per-trace missing from binary header",
            f"Binary header says {b['samples_per_trace']} samples per trace; the file layout proves {ns} (from {ns_src}).",
            fix=f"The fixed file declares {ns} samples per trace.",
            explain="Every trace has a fixed number of time samples. If the header states the wrong count, a loader "
            "slices the file in the wrong places and every trace after the first is garbage.",
            auto_fixed=True,
        ))
    if first_ns[endian] not in (ns, 0):
        issues.append(Issue(
            "NS_MISMATCH_TRACE", "warning", "Trace headers carry a different sample count",
            f"The first trace header (bytes 115-116) says {first_ns[endian]} samples, but the file layout proves {ns}. "
            "This is typical of files that were cropped in time without updating the trace headers.",
            fix=f"The fixed file writes {ns} into every trace header.",
            explain="Some loaders read the per-trace sample count instead of the file-level one and will mis-read this file.",
            auto_fixed=True,
        ))
    if rem and brk is None:
        issues.append(Issue(
            "TRUNCATED", "error", "File ends with a partial trace",
            f"After {n} complete traces there are {rem} bytes left over (a complete trace is {tb} bytes). "
            "The file was probably truncated during copying or tape transcription.",
            fix="The incomplete last trace is dropped from the fixed file. Check the delivery for a missing piece.",
            explain="The file is cut off part-way through its last trace, like a document missing its last page.",
            auto_fixed=True,
        ))

    dt = b["sample_interval"]
    if dt <= 0:
        dt = first_dt
        issues.append(Issue(
            "DT_MISSING_BINARY", "warning", "Sample interval missing from binary header",
            f"Binary header bytes 3217-3218 are zero; using the trace-header value {first_dt} microseconds.",
            fix="The fixed file declares the sample interval in the binary header.", auto_fixed=True,
            explain="The time between samples isn't recorded where loaders look for it.",
        ))
    elif first_dt and first_dt != dt:
        issues.append(Issue(
            "DT_MISMATCH", "warning", "Sample interval differs between binary and trace headers",
            f"Binary header says {dt} us, first trace header says {first_dt} us. The binary header value is used.",
            fix="The fixed file writes the same interval everywhere.", auto_fixed=True,
            explain="Two places in the file disagree on the time between samples.",
        ))

    lay = Layout(path, size, False, endian, enc, text, lines, b, n_ext, ext_text, doff, n_extra, fc, ns, float(dt), tb,
                 n, rem, None, issues, notes)
    return lay


def _structure_break(mm, size: int, doff: int, tb: int, n: int, endian: str):
    """Find where predicted trace headers stop being consistent. Returns (first_bad_index, kind) or None."""
    if n < 3:
        return None
    ns0 = _i16(mm, doff + 114, endian) & 0xFFFF
    seq0 = (_i32(mm, doff, endian), _i32(mm, doff + 4, endian))

    def ok(i: int) -> bool:
        off = doff + i * tb
        if off + TH > size:
            return False
        if ns0:
            v = _i16(mm, off + 114, endian) & 0xFFFF
            if v == ns0:
                return True
            if v == 0 and not any(bytes(mm[off : off + TH])):
                return True  # a blank header on an otherwise normal trace is not a structural break
            return False
        s1, s5 = _i32(mm, off, endian), _i32(mm, off + 4, endian)
        return s1 == seq0[0] + i or s5 == seq0[1] + i
    if not ns0 and not (_i32(mm, doff + tb, endian) == seq0[0] + 1 or _i32(mm, doff + tb + 4, endian) == seq0[1] + 1):
        return None  # nothing reliable to test against
    if ok(n - 1):
        return None
    lo, hi = 0, n - 1  # ok(lo) is True, ok(hi) is False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    k = hi
    # a real break persists: no trace after it may look normal again
    probe_after = sorted({min(n - 1, k + d) for d in (1, 2, 3, 5, 10, 50, 200)} | set(np.linspace(k, n - 1, 12).astype(int).tolist()))
    if any(ok(j) for j in probe_after if j > k):
        return None
    off = doff + k * tb
    blk = bytes(mm[off : min(size, off + TEXT_SIZE)])
    enc = decode_text_header(blk)[0] if len(blk) >= 400 else "binary/unreadable"
    if enc in ("ascii", "ebcdic"):
        if size - off <= 64 * TEXT_SIZE and size - off < 0.05 * size:
            return k, "trailer"
        return k, "second_file"
    return k, "break"


def _count_ext_text_variable(mm, size: int) -> int:
    n = 0
    while FILE_HEADER + (n + 1) * TEXT_SIZE <= size and n < 1000:
        blk = bytes(mm[FILE_HEADER + n * TEXT_SIZE : FILE_HEADER + (n + 1) * TEXT_SIZE])
        n += 1
        for enc in ("cp037", "latin-1"):
            if re.search(r"\(\(\s*SEG\s*:\s*EndText\s*\)\)", blk.decode(enc, errors="replace"), re.I):
                return n
    return 0


def _walk_variable(mm, size, doff, bps, endian, n_extra):
    offs, nss = [], []
    off = doff
    hb = TH * (1 + n_extra)
    while off + hb <= size:
        ns = _i16(mm, off + 114, endian) & 0xFFFF
        if ns == 0 or ns > 100000:
            return None
        offs.append(off)
        nss.append(ns)
        off += hb + ns * bps
        if len(offs) > 50_000_000:
            return None
    if off != size or len(offs) < 1:
        return None
    return np.asarray(offs, dtype=np.int64), np.asarray(nss, dtype=np.int64)


def _probe_su(mm, size: int, path: str) -> Layout | None:
    for endian in ("<", ">"):
        ns = _i16(mm, 114, endian) & 0xFFFF
        if not 0 < ns <= 50000:
            continue
        tb = TH + 4 * ns
        n, rem = divmod(size, tb)
        if n >= 1 and rem == 0 and _header_coherence(mm, size, 0, tb, n, ns, endian) > 0.8:
            dt = _i16(mm, 116, endian) & 0xFFFF
            lay = Layout(path, size, True, endian, "none", "", [""] * 40, {k: 0 for k in BIN}, 0, [], 0, 0, 5, ns,
                         float(dt), tb, n, 0, None, [], [])
            lay.issues.append(Issue(
                "SEISMIC_UNIX", "info", "This is a Seismic Unix (SU) file, not SEG-Y",
                "No 3600-byte file header; the file starts directly with a trace header. "
                f"Byte order {'little' if endian == '<' else 'big'}-endian, IEEE float samples.",
                fix="The fixed output is a standard SEG-Y file with a generated textual and binary header.",
                explain="Seismic Unix is an academic processing package that uses SEG-Y traces without the file header. "
                "Most commercial software can't read it directly.",
                auto_fixed=True,
            ))
            return lay
    return None


# ----------------------------------------------------------------------------- access
class TraceReader:
    """Random access to trace headers and samples of a probed file."""

    def __init__(self, layout: Layout):
        self.lay = layout
        self.mm = _mm(layout.path)
        lay = layout
        if lay.trace_offsets is None:
            end = lay.data_offset + lay.n_traces * lay.trace_bytes
            self.block = self.mm[lay.data_offset : end].reshape(lay.n_traces, lay.trace_bytes)
        else:
            self.block = None

    def headers(self, idx: np.ndarray | slice) -> np.ndarray:
        """Return uint8 array (k, 240) of the primary trace headers for the given trace indices."""
        lay = self.lay
        if self.block is not None:
            return np.asarray(self.block[idx, :TH])
        offs = lay.trace_offsets[idx]
        offs = np.atleast_1d(offs)
        return np.stack([np.asarray(self.mm[o : o + TH]) for o in offs]) if len(offs) else np.zeros((0, TH), np.uint8)

    def raw_samples(self, idx: np.ndarray | slice, fmt: int | None = None) -> np.ndarray:
        lay = self.lay
        fmt = fmt or lay.format_code
        width = FORMATS[fmt][0]
        nbytes = lay.ns * width
        if self.block is not None:
            start = lay.header_bytes
            return np.asarray(self.block[idx, start : start + nbytes])
        offs = np.atleast_1d(lay.trace_offsets[idx])
        out = np.zeros((len(offs), nbytes), np.uint8)
        for k, o in enumerate(offs):
            ns_i = _i16(self.mm, int(o) + 114, lay.endian) & 0xFFFF
            seg = np.asarray(self.mm[o + lay.header_bytes : o + lay.header_bytes + min(ns_i * width, nbytes)])
            out[k, : seg.size] = seg
        return out

    def samples(self, idx: np.ndarray | slice, fmt: int | None = None, endian: str | None = None) -> np.ndarray:
        fmt = fmt or self.lay.format_code
        raw = self.raw_samples(idx, fmt)
        return decode_samples(raw, fmt, endian or self.lay.endian)

    def field(self, idx, byte: int, typ: str, endian: str | None = None) -> np.ndarray:
        """Read one header field (1-based byte, 'int32'/'int16'/'float32') for the given traces."""
        endian = endian or self.lay.endian
        size = 2 if typ == "int16" else 4
        o = byte - 1
        if self.block is not None:
            raw = np.ascontiguousarray(self.block[idx, o : o + size])
        else:
            raw = np.ascontiguousarray(self.headers(idx)[:, o : o + size])
        code = {"int32": "i4", "int16": "i2", "float32": "f4"}[typ]
        return raw.view(np.dtype(endian + code)).reshape(-1)

    def fields_chunked(self, specs: list[tuple[int, str]], chunk: int = 200_000, progress=None) -> list[np.ndarray]:
        """Read several header fields for *every* trace, in chunks (memory-safe on huge files)."""
        n = self.lay.n_traces
        outs = [np.empty(n, dtype=np.float64 if t == "float32" else np.int64) for _, t in specs]
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            h = self.headers(slice(s, e)) if self.block is not None else self.headers(np.arange(s, e))
            for k, (byte, typ) in enumerate(specs):
                size = 2 if typ == "int16" else 4
                code = {"int32": "i4", "int16": "i2", "float32": "f4"}[typ]
                outs[k][s:e] = np.ascontiguousarray(h[:, byte - 1 : byte - 1 + size]).view(np.dtype(self.lay.endian + code)).reshape(-1)
            if progress:
                progress(e / n)
        return outs


def sample_indices(n: int, budget: int = 6000, block: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Pick trace indices for header analysis: contiguous blocks spread through the file.

    Contiguous blocks matter: patterns like 'inline stays constant while crossline counts up'
    can only be seen in consecutive traces. Returns (indices, block_id per index).
    """
    if n <= budget:
        return np.arange(n), np.zeros(n, dtype=np.int64)
    nblocks = max(2, budget // block)
    starts = np.linspace(0, n - block, nblocks).astype(np.int64)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    bid = np.repeat(np.arange(nblocks), block)
    idx, first = np.unique(idx, return_index=True)
    return idx, bid[first]
