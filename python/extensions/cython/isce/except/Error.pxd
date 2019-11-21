# error handling stub defined by CyError
cdef extern from "except/CyError.h":
    void raisePyError();
