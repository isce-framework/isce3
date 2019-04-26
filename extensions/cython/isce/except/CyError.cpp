#include <Python.h>
#include <isce/except/Error.h>

extern PyObject* errobj;

template<typename T>
void buildError(isce::except::Error<T> err) {
    auto tmp = Py_BuildValue("((s, i, s), s)",
                             err.info.file,
                             err.info.line,
                             err.info.func,
                             err.what());
    PyErr_SetObject(errobj, tmp);
}

void error() {
    using namespace isce::except;
    try {
        throw;
    } catch (LengthError err) {
        buildError(err);
    } catch (RuntimeError err) {
        buildError(err);
    }
}
