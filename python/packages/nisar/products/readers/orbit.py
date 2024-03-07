import xml.etree.ElementTree as ET
from isce3.core import DateTime, StateVector, Orbit
import journal


def load_orbit_from_xml(f, epoch: DateTime = None) -> Orbit:
    """Load orbit from XML file.

    Parameters
    ----------
    f : str, Path, or file
        File name or object containing XML data.
    epoch : isce3.core.DateTime, optional
        Desired reference epoch, defaults to time of first state vector.

    Returns
    -------
    orbit : isce3.core.Orbit
        Orbit object

    Notes
    -----
    File format is described in NISAR Orbit Ephemeris Product SIS, JPL D-102253.
    It contains information such as covariance estimates and maneuvers that
    are not parsed or represented in the output object.
    """
    root = ET.parse(f).getroot()
    svl = root.find("orbitStateVectorList")
    if svl is None:
        raise IOError("Could not parse orbit XML file.")
    n = int(svl.attrib["count"])
    states = []
    for node in svl.findall("orbitStateVector"):
        t = DateTime(node.find("utc").text)
        x = [float(node.find(name).text) for name in "xyz"]
        v = [float(node.find(name).text) for name in ("vx", "vy", "vz")]
        states.append(StateVector(t, x, v))
    if len(states) != n:
        raise IOError(f"Expected {n} orbit state vectors, got {len(states)}")
    if epoch is None:
        warning_channel = journal.warning("orbit.load_orbit_from_xml")
        warning_channel.log("No reference epoch provided. Using first date time "
                            "from XML file as orbit reference epoch.")
        epoch = states[0].datetime
    return Orbit(states, epoch)
