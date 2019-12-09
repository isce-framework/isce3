#cython: language_level=3

from cython.operator cimport dereference as deref
from cython.operator cimport address
from libcpp.vector cimport vector

from Cartesian cimport Vec3
from Orbit cimport Orbit, OrbitInterpBorderMode, OrbitInterpMethod, saveOrbitToH5, loadOrbitFromH5
from StateVector cimport StateVector

cdef class pyOrbit:
    cdef Orbit c_orbit

    def __cinit__(self):
        self.c_orbit = Orbit()

    def __init__(self, statevecs=None, reference_epoch=None, interp_method=None):
        # reference epoch defaults to first state vector
        if statevecs is not None and reference_epoch is None:
            reference_epoch = statevecs[0].datetime

        # set reference epoch before statevecs for accuracy
        if reference_epoch is not None:
            self.referenceEpoch = reference_epoch

        if statevecs is not None:
            self.setStateVectors(statevecs)

        if interp_method is None:
            interp_method = "Hermite"
        self.interpMethod = interp_method

    def getStateVectors(self):
        cdef vector[StateVector] c_statevecs = self.c_orbit.getStateVectors()
        statevecs = []
        for i in range(self.size):
            sv = pyStateVector()
            sv.c_statevector = c_statevecs[i]
            statevecs.append(sv)
        return statevecs

    def setStateVectors(self, statevecs):
        cdef vector[StateVector] c_statevecs
        cdef pyStateVector sv
        for _sv in statevecs:
            sv = _sv
            c_statevecs.push_back(sv.c_statevector)
        self.c_orbit.setStateVectors(c_statevecs)

    @property
    def referenceEpoch(self):
        cdef pyDateTime reference_epoch = pyDateTime()
        reference_epoch.c_datetime[0] = self.c_orbit.referenceEpoch()
        return reference_epoch

    @referenceEpoch.setter
    def referenceEpoch(self, pyDateTime reference_epoch):
        self.c_orbit.referenceEpoch(deref(reference_epoch.c_datetime))

    @property
    def interpMethod(self):
        if self.c_orbit.interpMethod() == OrbitInterpMethod.Hermite:
            return "Hermite"
        if self.c_orbit.interpMethod() == OrbitInterpMethod.Legendre:
            return "Legendre"
        raise RuntimeError("unrecognized orbit interpolation method")

    @interpMethod.setter
    def interpMethod(self, method):
        if method == "Hermite":
            self.c_orbit.interpMethod(OrbitInterpMethod.Hermite)
        elif method == "Legendre":
            self.c_orbit.interpMethod(OrbitInterpMethod.Legendre)
        else:
            raise ValueError("unknown orbit interpolation method '" + str(method) + "'")

    @property
    def startTime(self):
        return self.c_orbit.startTime()

    @property
    def midTime(self):
        return self.c_orbit.midTime()

    @property
    def endTime(self):
        return self.c_orbit.endTime()

    @property
    def startDateTime(self):
        cdef pyDateTime dt = pyDateTime()
        dt.c_datetime[0] = self.c_orbit.startDateTime()
        return dt

    @property
    def midDateTime(self):
        cdef pyDateTime dt = pyDateTime()
        dt.c_datetime[0] = self.c_orbit.midDateTime()
        return dt

    @property
    def endDateTime(self):
        cdef pyDateTime dt = pyDateTime()
        dt.c_datetime[0] = self.c_orbit.endDateTime()
        return dt

    @property
    def spacing(self):
        return self.c_orbit.spacing()

    @property
    def size(self):
        return self.c_orbit.size()

    def time(self, i):
        return self.c_orbit.time(i)

    def position(self, i):
        cdef Vec3 c_position = self.c_orbit.position(i)
        return (c_position[0], c_position[1], c_position[2])

    def velocity(self, i):
        cdef Vec3 c_velocity = self.c_orbit.velocity(i)
        return (c_velocity[0], c_velocity[1], c_velocity[2])

    def interpolate(self, t, border_mode=None):
        if border_mode is None:
            border_mode = "Error"

        cdef Vec3 c_position = Vec3()
        cdef Vec3 c_velocity = Vec3()

        if border_mode == "Error":
            self.c_orbit.interpolate(address(c_position), address(c_velocity), t, OrbitInterpBorderMode.Error)
        elif border_mode == "Extrapolate":
            self.c_orbit.interpolate(address(c_position), address(c_velocity), t, OrbitInterpBorderMode.Extrapolate)
        elif border_mode == "FillNaN":
            self.c_orbit.interpolate(address(c_position), address(c_velocity), t, OrbitInterpBorderMode.FillNaN)
        else:
            raise ValueError("unknown orbit interpolation border mode '" + str(border_mode) + "'")

        return (c_position[0], c_position[1], c_position[2]), (c_velocity[0], c_velocity[1], c_velocity[2])

    def saveToH5(self, group):
        cdef hid_t groupid = group.id.id
        cdef IGroup c_igroup = IGroup(groupid)
        saveOrbitToH5(c_igroup, self.c_orbit)

    @classmethod
    def loadFromH5(cls, group):
        cdef hid_t groupid = group.id.id
        cdef IGroup c_igroup = IGroup(groupid)
        orbit = pyOrbit()
        loadOrbitFromH5(c_igroup, orbit.c_orbit)
        return orbit

    def __eq__(self, pyOrbit other):
        return self.c_orbit == other.c_orbit

    def __ne__(self, pyOrbit other):
        return self.c_orbit != other.c_orbit
