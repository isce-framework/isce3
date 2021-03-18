import xml.etree.ElementTree as ET
from isce3.core import DateTime, Quaternion, Attitude


def _require(node: ET.Element, pattern: str) -> ET.Element:
    """Like Element.find but error if no match is found.
    """
    x = node.find(pattern)
    if x is None:
        raise IOError(f"Could not find XML element '{pattern}'")
    return x


def load_attitude_from_xml(f, epoch: DateTime = None, band="L") -> Attitude:
    """Load attitude from XML file.

    Parameters
    ----------
    f : str, Path, or file
        File name or object containing XML data.
    epoch : isce3.core.DateTime, optional
        Desired reference epoch, defaults to time of first state vector.
    band : {'L', 'S'}, optional
        Which frequency band to read, default to 'L'.

    Returns
    -------
    attitude : isce3.core.Attitude
        Attitude object.

    Notes
    -----
    File format is described in NISAR Radar Pointing Product SIS, JPL D-102264.
    It contains information that may not be parsed or represented in the
    output object.
    """
    if band not in ("L", "S"):
        raise ValueError(f"Expected band in {{'L', 'S'}} got {band}")
    root = ET.parse(f).getroot()  # radarPointing
    svl = _require(root, f"{band}SAR/radarPointingStateVectorList")
    n = int(svl.attrib["count"])
    if n <= 0:
        raise ValueError(f"Need multiple quaternions, got count={n}")
    datetimes, qs = [], []
    for node in svl.findall("radarPointingStateVector"):
        t = DateTime(_require(node, "utc").text)
        q = (float(_require(node, name).text)
             for name in ("q0", "q1", "q2", "q3"))
        datetimes.append(t)
        qs.append(Quaternion(*q))
    if len(datetimes) != n:
        raise IOError(f"Expected {n} attitude entries, got {len(datetimes)}")
    if epoch is None:
        epoch = datetimes[0]
    # NOTE Assume times are sorted and let Attitude ctor throw error if not.
    times = [(t - epoch).total_seconds() for t in datetimes]
    return Attitude(times, qs, epoch)
