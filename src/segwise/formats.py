"""Sample formats used in SEG-Y and fast vectorised converters.

SEG-Y stores trace samples in one of a handful of numeric encodings, named by
the "data sample format code" in the binary file header (bytes 3225-3226).
The one that causes the most trouble is code 1, IBM System/360 floating point,
which pre-dates IEEE 754 and is still the default in most legacy data.
"""
from __future__ import annotations

import numpy as np

# code -> (bytes per sample, numpy kind, human name)
FORMATS: dict[int, tuple[int, str, str]] = {
    1: (4, "ibm", "4-byte IBM floating point"),
    2: (4, "i4", "4-byte two's complement integer"),
    3: (2, "i2", "2-byte two's complement integer"),
    4: (4, "fixgain", "4-byte fixed point with gain (obsolete)"),
    5: (4, "f4", "4-byte IEEE floating point"),
    6: (8, "f8", "8-byte IEEE floating point"),
    7: (3, "i3", "3-byte two's complement integer"),
    8: (1, "i1", "1-byte two's complement integer"),
    9: (8, "i8", "8-byte two's complement integer"),
    10: (4, "u4", "4-byte unsigned integer"),
    11: (2, "u2", "2-byte unsigned integer"),
    12: (8, "u8", "8-byte unsigned integer"),
    15: (3, "u3", "3-byte unsigned integer"),
    16: (1, "u1", "1-byte unsigned integer"),
}

# Formats that share a byte width can be confused with one another; these are
# the alternatives we test when checking whether the declared format is right.
SAME_WIDTH = {w: [c for c, (bw, _, _) in FORMATS.items() if bw == w] for w in (1, 2, 3, 4, 8)}
# the 'normal' formats we will actually propose as alternatives
COMMON = [1, 5, 2, 3, 8, 6]


def bytes_per_sample(code: int) -> int:
    return FORMATS[code][0]


def format_name(code: int) -> str:
    return FORMATS.get(code, (0, "", f"unknown format code {code}"))[2]


def ibm_to_float(words: np.ndarray) -> np.ndarray:
    """Convert IBM System/360 single-precision words (as uint32, native order) to float32.

    IBM float: 1 sign bit, 7-bit base-16 exponent biased by 64, 24-bit fraction
    with no hidden bit:  value = (-1)^s * 0.fraction * 16^(exponent - 64).
    Values outside the IEEE single range become +/-inf (callers flag these).
    """
    u = np.asarray(words, dtype=np.uint32)
    sign = (u >> np.uint32(31)).astype(bool)
    exponent = ((u >> np.uint32(24)) & np.uint32(0x7F)).astype(np.int32)
    fraction = (u & np.uint32(0x00FFFFFF)).astype(np.float64)
    # 0.fraction * 16^(e-64) == fraction * 2^(4*(e-64) - 24)
    with np.errstate(over="ignore"):
        value = np.ldexp(fraction, 4 * (exponent - 64) - 24)
        value[sign] *= -1.0
        return value.astype(np.float32)


def float_to_ibm(values: np.ndarray) -> np.ndarray:
    """Convert float32/float64 values to IBM single-precision words (uint32, native order)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.uint32)
    nz = (v != 0) & np.isfinite(v)
    if not nz.any():
        return out
    a = np.abs(v[nz])
    sign = (v[nz] < 0).astype(np.uint32) << np.uint32(31)
    # find exponent e such that 16^(e-64-1) <= a < 16^(e-64)
    e = np.floor(np.log2(a) / 4.0).astype(np.int64) + 65
    frac = a / np.power(16.0, (e - 64).astype(np.float64))
    # normalise against rounding at the boundaries
    hi = frac >= 1.0
    e[hi] += 1
    frac[hi] /= 16.0
    lo = frac < 1.0 / 16.0
    e[lo] -= 1
    frac[lo] *= 16.0
    mant = np.round(frac * 2**24).astype(np.int64)
    carry = mant >= 2**24
    mant[carry] >>= 4
    e[carry] += 1
    under = e < 0
    over = e > 127
    mant[under] = 0
    e[under] = 0
    mant[over] = 0xFFFFFF
    e[over] = 127
    words = sign | (e.astype(np.uint32) << np.uint32(24)) | mant.astype(np.uint32)
    words[under] = 0
    out[nz] = words
    return out


def decode_samples(raw: np.ndarray, code: int, endian: str) -> np.ndarray:
    """Decode a uint8 array shaped (..., ns*bps) into float32 samples shaped (..., ns)."""
    width, kind, _ = FORMATS[code]
    raw = np.ascontiguousarray(raw)
    if kind == "ibm":
        words = raw.view(np.dtype(endian + "u4"))
        return ibm_to_float(words.astype(np.uint32))
    if kind in ("i3", "u3"):
        b = raw.reshape(raw.shape[:-1] + (-1, 3)).astype(np.int32)
        if endian == "<":
            b = b[..., ::-1]
        v = (b[..., 0] << 16) | (b[..., 1] << 8) | b[..., 2]
        if kind == "i3":
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
        return v.astype(np.float32)
    if kind == "fixgain":
        b = raw.reshape(raw.shape[:-1] + (-1, 4))
        if endian == "<":
            b = b[..., ::-1]
        gain = b[..., 1].astype(np.int32)
        mant = (b[..., 2].astype(np.int32) << 8 | b[..., 3].astype(np.int32)).astype(np.int16)
        return np.ldexp(mant.astype(np.float64), -gain).astype(np.float32)
    dt = np.dtype(kind if width == 1 else endian + kind)
    with np.errstate(over="ignore", invalid="ignore"):
        return raw.view(dt).astype(np.float32)


def encode_ieee(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=">f4").tobytes()


def swap_sample_bytes(raw: np.ndarray, code: int) -> np.ndarray:
    """Byte-swap raw sample bytes (uint8, last axis = ns*bps) for formats wider than one byte."""
    width = FORMATS[code][0]
    if width == 1:
        return raw
    shape = raw.shape
    r = np.ascontiguousarray(raw).reshape(-1, width)[:, ::-1]
    return np.ascontiguousarray(r).reshape(shape)
