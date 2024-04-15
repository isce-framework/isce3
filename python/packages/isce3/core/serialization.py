from __future__ import annotations
from isce3.core import DateTime, Orbit, StateVector, TimeDelta
import h5py

def parse_iso_date(iso_date: str) -> DateTime:
    """
    Parse ISO date string to DateTime

    This is a direct translation from C++ code in isce3,
    and could probably be made more robust/intuitive.
    """
    utc_ref = ""
    pos_iso = iso_date.rfind('-')
    if pos_iso != -1:
        utc_ref = iso_date[pos_iso - 7:]
        # remove any trailing whitespace
        utc_ref = utc_ref.rstrip(' ')
    return DateTime(utc_ref)

def load_orbit_from_h5_group(group: h5py.Group) -> Orbit:
    """
    Load orbit data from a group in an HDF5 file.

    The organization of orbit data in the group is assumed to conform to
    NISAR product specifications.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group containing the orbit metadata.

    Returns
    -------
    orbit : Orbit
        The orbit data.
    """
    time = group['time']
    pos = group['position']
    vel = group['velocity']
    orbit_kwargs = {}
    if 'orbitType' in group:
        # Strip trailing null characters as a bandaid for handling older datasets.
        orbit_type = group['orbitType'][...].tobytes().decode().rstrip('\x00')
        orbit_kwargs['type'] = orbit_type

    if time.ndim != 1 or pos.ndim != 2 or vel.ndim != 2:
        raise ValueError("unexpected orbit state vector dims")

    if pos.shape[1] != 3 or vel.shape[1] != 3:
        raise ValueError("unexpected orbit position/velocity vector size")

    size = time.shape[0]
    if pos.shape[0] != size or vel.shape[0] != size:
        raise ValueError("mismatched orbit state vector component sizes")

    # get reference epoch
    unit_attr = time.attrs['units']
    # result may be str or bytes, convert to str if needed
    if type(unit_attr) is not str:
        unit_attr = unit_attr.decode('utf-8')
    epoch = parse_iso_date(unit_attr)

    # convert to state vectors
    statevecs = [StateVector(epoch + TimeDelta(time[i]), pos[i], vel[i])
                 for i in range(size)]

    # construct Orbit object
    result = Orbit(statevecs, epoch, **orbit_kwargs)

    # set interpolation method, if specified
    if 'interpMethod' in group.keys():
        result.set_interp_method(group['interpMethod'][()])

    return result
