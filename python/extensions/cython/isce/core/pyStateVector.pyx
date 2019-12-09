#cython: language_level=3

from StateVector cimport StateVector

cdef class pyStateVector:
    cdef StateVector c_statevector

    def __cinit__(self):
        self.c_statevector = StateVector()

    def __init__(self, datetime=None, position=None, velocity=None):
        if datetime is not None:
            self.datetime = datetime
        if position is not None:
            self.position = position
        if velocity is not None:
            self.velocity = velocity

    @property
    def datetime(self):
        return pyDateTime(self.c_statevector.datetime.isoformat().decode("UTF-8"))

    @datetime.setter
    def datetime(self, pyDateTime datetime):
        self.c_statevector.datetime.strptime(pyStringToBytes(datetime.isoformat()))

    @property
    def position(self):
        x = self.c_statevector.position[0]
        y = self.c_statevector.position[1]
        z = self.c_statevector.position[2]
        return  (x, y, z)

    @position.setter
    def position(self, position):
        self.c_statevector.position[0] = position[0]
        self.c_statevector.position[1] = position[1]
        self.c_statevector.position[2] = position[2]

    @property
    def velocity(self):
        vx = self.c_statevector.velocity[0]
        vy = self.c_statevector.velocity[1]
        vz = self.c_statevector.velocity[2]
        return  (vx, vy, vz)

    @velocity.setter
    def velocity(self, velocity):
        self.c_statevector.velocity[0] = velocity[0]
        self.c_statevector.velocity[1] = velocity[1]
        self.c_statevector.velocity[2] = velocity[2]
