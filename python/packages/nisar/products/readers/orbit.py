import xml.etree.ElementTree as ET
import isce3
from isce3.core import DateTime, StateVector, Orbit, crop_external_orbit
import journal


def load_orbit(nisar_product, orbit_file, orbit_file_ref_epoch=None):
    '''
    Load the orbit from a NISAR product object or from an external file
    (if provided)

    Parameters
    ----------
    nisar_product: nisar.products.readers.Base
        NISAR product object (e.g., RSLC) containing orbit ephemeris
    orbit_file: str or None
        Optional external orbit file
    orbit_file_ref_epoch: isce3.core.DateTime or None
        Optional reference epoch to be used as a reference for external orbit
        files (e.g., the reference epoch from the radar grid). If `None`,
        the reference epoch for the input NISAR product will be used instead

    Returns
    -------
    orbit: isce3.core.Orbit
        ISCE3 orbit object containing orbit ephemeris
    '''

    warning_channel = journal.warning("load_orbit")

    # load the orbit from the RSLC metadata
    orbit = nisar_product.getOrbit()

    if orbit_file_ref_epoch is None:
        orbit_file_ref_epoch = orbit.reference_epoch

    # if an external orbit file has been provided, load it
    # based on existing orbit within `nisar_product`
    if orbit_file is not None:
        external_orbit = load_orbit_from_xml(orbit_file,
                                             orbit_file_ref_epoch)

        # Apply 2 mins of padding before / after sensing period when
        # cropping the external orbit.
        # 2 mins of margin is based on the number of IMAGEN TEC samples
        # required for TEC computation, with few more safety margins for
        # possible needs in the future.
        #
        # `7` in the line below is came from the default value for `npad`
        # in `crop_external_orbit()`. See:
        # .../isce3/python/isce3/core/crop_external_orbit.py
        npad = max(int(120.0 / external_orbit.spacing), 7)
        orbit = crop_external_orbit(external_orbit, orbit, npad=npad)

    elif orbit.reference_epoch != orbit_file_ref_epoch:
        warning_channel.log('The reference epoch provided to load_orbit() does'
                            ' not match with the orbit reference epoch of the'
                            ' input NISAR product. The orbit reference epoch'
                            ' of the NISAR product will be updated to match'
                            ' the input reference epoch.')
        orbit.update_reference_epoch(orbit_file_ref_epoch)

    # ensure that the orbit reference epoch has not fractional part
    # otherwise, trancate it to seconds precision
    orbit_reference_epoch = orbit.reference_epoch
    if orbit_reference_epoch.frac != 0:
        warning_channel.log('the orbit reference epoch is not an'
                            ' integer number. Truncating it'
                            ' to seconds precision and'
                            ' updating the orbit ephemeris'
                            ' accordingly.')

        epoch = isce3.core.DateTime(orbit_reference_epoch.year,
                                    orbit_reference_epoch.month,
                                    orbit_reference_epoch.day,
                                    orbit_reference_epoch.hour,
                                    orbit_reference_epoch.minute,
                                    orbit_reference_epoch.second)

        orbit.update_reference_epoch(epoch)

    return orbit


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

    warning_channel = journal.warning("orbit.load_orbit_from_xml")

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
        warning_channel.log("No reference epoch provided. Using first date time "
                            "from XML file as orbit reference epoch.")
        epoch = states[0].datetime

    orbit_kwargs = {}

    # read the orbit ephemeris precision type
    # first, try to read it from `productInformation/productType`
    orbit_type_et = root.find('productInformation/productType')
    if orbit_type_et is None:

        # if not found, try to read it from `productInformation/fileType`
        orbit_type_et = root.find('productInformation/fileType')

    # if still not found, raise a warning
    if orbit_type_et is None:
        warning_channel.log("Orbit file does not contain precision"
                            ' type (e.g., "FOE", "NOE", "MOE", "POE", or'
                            ' "Custom").')

    # otherwise, add the precision type to the orbit kwargs
    else:
        orbit_kwargs['type'] = orbit_type_et.text

    # create Orbit object
    orbit = Orbit(states, epoch, **orbit_kwargs)

    return orbit
