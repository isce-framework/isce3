
import numpy as np
import isce3


def shift_orbit(orbit_obj, offset_x, offset_y):
    """
    Shift orbit in horizontal or vertical coordinates, or both.

    Parameters
    ----------
    orbit_obj : isce3.core.Orbit
        The orbit object
    offset_x : scalar
        Horizontal offset in meters and in the cross-track direction
    offset_y : scalar
        Vertical offset upwards in meters

    Returns
    -------
    isce3.core.Orbit
        The shifted orbit object
    """

    velocity_vector = orbit_obj.velocity
    right_direction = np.cross(velocity_vector, orbit_obj.position)

    # Normalize normal vectors
    right_direction_unit = (
        right_direction /
        np.linalg.norm(right_direction, axis=1)[:, np.newaxis])

    # Apply the offset
    up_direction_unit = (
        orbit_obj.position /
        np.linalg.norm(orbit_obj.position, axis=1)[:, np.newaxis])

    pos = (orbit_obj.position + offset_x * right_direction_unit +
           offset_y * up_direction_unit)

    time_period = 1

    sv_list = []
    for i in range(orbit_obj.size):
        t = orbit_obj.reference_epoch + isce3.core.TimeDelta(orbit_obj.time[i])

        if i == 0:
            pos_after, _ = orbit_obj.interpolate(orbit_obj.time[i] +
                                                 time_period)
            pos_before, _ = orbit_obj.interpolate(orbit_obj.time[i])

        elif i < orbit_obj.size - 1:
            pos_after, _ = orbit_obj.interpolate(orbit_obj.time[i] +
                                                 time_period / 2)
            pos_before, _ = orbit_obj.interpolate(orbit_obj.time[i] -
                                                  time_period / 2)

        else:
            pos_after, _ = orbit_obj.interpolate(orbit_obj.time[i])
            pos_before, _ = orbit_obj.interpolate(orbit_obj.time[i] -
                                                  time_period)

        vel = (np.asarray(pos_after) - np.asarray(pos_before)) / time_period

        sv = isce3.core.StateVector(t, pos[i, :], vel)
        sv_list.append(sv)

    shifted_orbit_obj = isce3.core.Orbit(sv_list, orbit_obj.reference_epoch,
                                         type=orbit_obj.get_type())

    return shifted_orbit_obj
