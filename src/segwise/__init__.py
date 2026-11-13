"""segwise - forensic SEG-Y loader.

Works out where the geometry, coordinates and coordinate reference system of a
SEG-Y file really are, proves each decision with evidence from the file, and
writes a clean standard file plus a load sheet that loads first time.

    from segwise.analyze import analyze
    report = analyze("survey.sgy")
"""

__version__ = "0.1.0"
