"""The SEG-Y rev 1 field tables: what each byte position is *supposed* to hold.

Byte numbers are 1-based and relative to the start of the header, exactly as
they are written in the SEG-Y standard and in every geophysicist's head
("inline is at 189"). Internally we convert to 0-based offsets.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    byte: int  # 1-based start byte within the header
    size: int  # 2 or 4 (or 8)
    name: str
    label: str  # plain-English description


TRACE_FIELDS: list[Field] = [
    Field(1, 4, "TRACE_SEQUENCE_LINE", "Trace sequence number within line"),
    Field(5, 4, "TRACE_SEQUENCE_FILE", "Trace sequence number within file"),
    Field(9, 4, "FieldRecord", "Original field record number (FFID)"),
    Field(13, 4, "TraceNumber", "Trace number within field record (channel)"),
    Field(17, 4, "EnergySourcePoint", "Energy source point number"),
    Field(21, 4, "CDP", "Ensemble number (CDP / CMP)"),
    Field(25, 4, "CDP_TRACE", "Trace number within ensemble"),
    Field(29, 2, "TraceIdentificationCode", "Trace identification code"),
    Field(31, 2, "NSummedTraces", "Number of vertically summed traces"),
    Field(33, 2, "NStackedTraces", "Number of horizontally stacked traces (fold)"),
    Field(35, 2, "DataUse", "Data use (1=production, 2=test)"),
    Field(37, 4, "offset", "Source-receiver offset"),
    Field(41, 4, "ReceiverGroupElevation", "Receiver group elevation"),
    Field(45, 4, "SourceSurfaceElevation", "Surface elevation at source"),
    Field(49, 4, "SourceDepth", "Source depth below surface"),
    Field(53, 4, "ReceiverDatumElevation", "Datum elevation at receiver"),
    Field(57, 4, "SourceDatumElevation", "Datum elevation at source"),
    Field(61, 4, "SourceWaterDepth", "Water depth at source"),
    Field(65, 4, "GroupWaterDepth", "Water depth at group"),
    Field(69, 2, "ElevationScalar", "Scalar for elevations and depths"),
    Field(71, 2, "SourceGroupScalar", "Scalar for coordinates"),
    Field(73, 4, "SourceX", "Source X coordinate"),
    Field(77, 4, "SourceY", "Source Y coordinate"),
    Field(81, 4, "GroupX", "Receiver group X coordinate"),
    Field(85, 4, "GroupY", "Receiver group Y coordinate"),
    Field(89, 2, "CoordinateUnits", "Coordinate units (1=length, 2=arcsec, 3=degrees, 4=DMS)"),
    Field(91, 2, "WeatheringVelocity", "Weathering velocity"),
    Field(93, 2, "SubWeatheringVelocity", "Subweathering velocity"),
    Field(95, 2, "SourceUpholeTime", "Uphole time at source (ms)"),
    Field(97, 2, "GroupUpholeTime", "Uphole time at group (ms)"),
    Field(99, 2, "SourceStaticCorrection", "Source static correction (ms)"),
    Field(101, 2, "GroupStaticCorrection", "Group static correction (ms)"),
    Field(103, 2, "TotalStaticApplied", "Total static applied (ms)"),
    Field(105, 2, "LagTimeA", "Lag time A (ms)"),
    Field(107, 2, "LagTimeB", "Lag time B (ms)"),
    Field(109, 2, "DelayRecordingTime", "Delay recording time (ms)"),
    Field(111, 2, "MuteTimeStart", "Mute time start (ms)"),
    Field(113, 2, "MuteTimeEND", "Mute time end (ms)"),
    Field(115, 2, "TRACE_SAMPLE_COUNT", "Number of samples in this trace"),
    Field(117, 2, "TRACE_SAMPLE_INTERVAL", "Sample interval for this trace (microseconds)"),
    Field(119, 2, "GainType", "Gain type of field instruments"),
    Field(121, 2, "InstrumentGainConstant", "Instrument gain constant (dB)"),
    Field(123, 2, "InstrumentInitialGain", "Instrument early or initial gain (dB)"),
    Field(125, 2, "Correlated", "Correlated (1=no, 2=yes)"),
    Field(127, 2, "SweepFrequencyStart", "Sweep frequency at start (Hz)"),
    Field(129, 2, "SweepFrequencyEnd", "Sweep frequency at end (Hz)"),
    Field(131, 2, "SweepLength", "Sweep length (ms)"),
    Field(133, 2, "SweepType", "Sweep type"),
    Field(135, 2, "SweepTraceTaperLengthStart", "Sweep taper length at start (ms)"),
    Field(137, 2, "SweepTraceTaperLengthEnd", "Sweep taper length at end (ms)"),
    Field(139, 2, "TaperType", "Taper type"),
    Field(141, 2, "AliasFilterFrequency", "Alias filter frequency (Hz)"),
    Field(143, 2, "AliasFilterSlope", "Alias filter slope (dB/octave)"),
    Field(145, 2, "NotchFilterFrequency", "Notch filter frequency (Hz)"),
    Field(147, 2, "NotchFilterSlope", "Notch filter slope (dB/octave)"),
    Field(149, 2, "LowCutFrequency", "Low-cut frequency (Hz)"),
    Field(151, 2, "HighCutFrequency", "High-cut frequency (Hz)"),
    Field(153, 2, "LowCutSlope", "Low-cut slope (dB/octave)"),
    Field(155, 2, "HighCutSlope", "High-cut slope (dB/octave)"),
    Field(157, 2, "YearDataRecorded", "Year data recorded"),
    Field(159, 2, "DayOfYear", "Day of year"),
    Field(161, 2, "HourOfDay", "Hour of day"),
    Field(163, 2, "MinuteOfHour", "Minute of hour"),
    Field(165, 2, "SecondOfMinute", "Second of minute"),
    Field(167, 2, "TimeBaseCode", "Time basis code"),
    Field(169, 2, "TraceWeightingFactor", "Trace weighting factor"),
    Field(171, 2, "GeophoneGroupNumberRoll1", "Geophone group number of roll switch position one"),
    Field(173, 2, "GeophoneGroupNumberFirstTraceOrigField", "Geophone group number of first trace"),
    Field(175, 2, "GeophoneGroupNumberLastTraceOrigField", "Geophone group number of last trace"),
    Field(177, 2, "GapSize", "Gap size (total number of groups dropped)"),
    Field(179, 2, "OverTravel", "Over travel associated with taper"),
    Field(181, 4, "CDP_X", "Ensemble (CDP) X coordinate"),
    Field(185, 4, "CDP_Y", "Ensemble (CDP) Y coordinate"),
    Field(189, 4, "INLINE_3D", "3D inline number"),
    Field(193, 4, "CROSSLINE_3D", "3D crossline number"),
    Field(197, 4, "ShotPoint", "Shotpoint number (2D)"),
    Field(201, 2, "ShotPointScalar", "Scalar for shotpoint number"),
    Field(203, 2, "TraceValueMeasurementUnit", "Trace value measurement unit"),
    Field(205, 4, "TransductionConstantMantissa", "Transduction constant mantissa"),
    Field(209, 2, "TransductionConstantPower", "Transduction constant exponent"),
    Field(211, 2, "TransductionUnit", "Transduction units"),
    Field(213, 2, "TraceIdentifier", "Device/trace identifier"),
    Field(215, 2, "ScalarTraceHeader", "Scalar for times in bytes 95-114"),
    Field(217, 2, "SourceType", "Source type/orientation"),
    Field(219, 4, "SourceEnergyDirectionMantissa", "Source energy direction (mantissa)"),
    Field(223, 2, "SourceEnergyDirectionExponent", "Source energy direction (exponent)"),
    Field(225, 4, "SourceMeasurementMantissa", "Source measurement mantissa"),
    Field(229, 2, "SourceMeasurementExponent", "Source measurement exponent"),
    Field(231, 2, "SourceMeasurementUnit", "Source measurement unit"),
    Field(233, 4, "UnassignedInt1", "Unassigned (free for vendor use)"),
    Field(237, 4, "UnassignedInt2", "Unassigned (free for vendor use)"),
]

TRACE_FIELD_BY_BYTE = {f.byte: f for f in TRACE_FIELDS}


def trace_field_at(byte: int) -> Field | None:
    """The standard field that *contains* this 1-based byte, if any."""
    for f in TRACE_FIELDS:
        if f.byte <= byte < f.byte + f.size:
            return f
    return None


# Binary file header: byte positions relative to the start of the FILE (3201-3600).
BIN = {
    "job_id": (3201, 4),
    "line_number": (3205, 4),
    "reel_number": (3209, 4),
    "traces_per_ensemble": (3213, 2),
    "aux_traces_per_ensemble": (3215, 2),
    "sample_interval": (3217, 2),
    "orig_sample_interval": (3219, 2),
    "samples_per_trace": (3221, 2),
    "orig_samples_per_trace": (3223, 2),
    "format_code": (3225, 2),
    "ensemble_fold": (3227, 2),
    "sorting_code": (3229, 2),
    "vertical_sum_code": (3231, 2),
    "measurement_system": (3255, 2),
    "impulse_polarity": (3257, 2),
    "vibratory_polarity": (3259, 2),
    "ext_samples_per_trace": (3269, 4),
    "byte_order_marker": (3297, 4),
    "rev_major": (3501, 1),
    "rev_minor": (3502, 1),
    "fixed_length_flag": (3503, 2),
    "n_ext_text_headers": (3505, 2),
    "max_extra_trace_headers": (3507, 4),
}

BIN_LABELS = {
    "job_id": "Job identification number",
    "line_number": "Line number",
    "reel_number": "Reel number",
    "traces_per_ensemble": "Data traces per ensemble",
    "aux_traces_per_ensemble": "Auxiliary traces per ensemble",
    "sample_interval": "Sample interval (microseconds)",
    "orig_sample_interval": "Original field sample interval",
    "samples_per_trace": "Samples per trace",
    "orig_samples_per_trace": "Original field samples per trace",
    "format_code": "Data sample format code",
    "ensemble_fold": "Ensemble fold",
    "sorting_code": "Trace sorting code",
    "vertical_sum_code": "Vertical sum code",
    "measurement_system": "Measurement system (1=metres, 2=feet)",
    "impulse_polarity": "Impulse signal polarity",
    "vibratory_polarity": "Vibratory polarity code",
    "ext_samples_per_trace": "Extended samples per trace (rev 2)",
    "byte_order_marker": "Byte order marker (rev 2: 16909060)",
    "rev_major": "SEG-Y revision (major)",
    "rev_minor": "SEG-Y revision (minor)",
    "fixed_length_flag": "Fixed length trace flag",
    "n_ext_text_headers": "Number of extended textual headers",
    "max_extra_trace_headers": "Additional trace headers per trace (rev 2)",
}

SORTING_CODES = {
    -1: "other", 0: "unknown", 1: "as recorded (no sorting)", 2: "CDP ensemble",
    3: "single fold continuous profile", 4: "horizontally stacked", 5: "common source point",
    6: "common receiver point", 7: "common offset point", 8: "common mid-point", 9: "common conversion point",
}

# Preferred byte positions when the same quantity is found in several places.
INLINE_PREFERENCE = [189, 9, 5, 221, 213, 17, 197, 225, 233, 237]
XLINE_PREFERENCE = [193, 21, 17, 13, 25, 225, 197, 233, 237]
COORD_PAIR_PREFERENCE = [(181, 185), (73, 77), (81, 85), (201, 205), (221, 225), (233, 237)]
