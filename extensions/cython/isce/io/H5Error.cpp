#include <string>

#include <Python.h>

#include <H5Cpp.h>
#include <H5Exception.h>

// Exception types
// https://support.hdfgroup.org/HDF5/doc/cpplus_RM/class_h5_1_1_exception.html

extern PyObject* h5exception;
extern PyObject* h5attributeiexception;
extern PyObject* h5datasetiexception;
extern PyObject* h5dataspaceiexception;
extern PyObject* h5datatypeiexception;
extern PyObject* h5fileiexception;
extern PyObject* h5groupiexception;
extern PyObject* h5idcomponentexception;
extern PyObject* h5libraryiexception;
extern PyObject* h5locationexception;
extern PyObject* h5objheaderiexception;
extern PyObject* h5proplistiexception;
extern PyObject* h5referenceexception;

void translateH5Exception() {
    try {
        throw;
    } catch (const H5::AttributeIException& e) {
        PyErr_SetString(h5attributeiexception, e.getCDetailMsg());
    } catch (const H5::DataSetIException& e) {
        PyErr_SetString(h5datasetiexception, e.getCDetailMsg());
    } catch (const H5::DataSpaceIException& e) {
        PyErr_SetString(h5dataspaceiexception, e.getCDetailMsg());
    } catch (const H5::DataTypeIException& e) {
        PyErr_SetString(h5datatypeiexception, e.getCDetailMsg());
    } catch (const H5::FileIException& e) {
        PyErr_SetString(h5fileiexception, e.getCDetailMsg());
    } catch (const H5::GroupIException& e) {
        PyErr_SetString(h5groupiexception, e.getCDetailMsg());
    } catch (const H5::IdComponentException& e) {
        PyErr_SetString(h5idcomponentexception, e.getCDetailMsg());
    } catch (const H5::LibraryIException& e) {
        PyErr_SetString(h5libraryiexception, e.getCDetailMsg());
    } catch (const H5::LocationException& e) {
        PyErr_SetString(h5locationexception, e.getCDetailMsg());
    } catch (const H5::ObjHeaderIException& e) {
        PyErr_SetString(h5objheaderiexception, e.getCDetailMsg());
    } catch (const H5::PropListIException& e) {
        PyErr_SetString(h5proplistiexception, e.getCDetailMsg());
    } catch (const H5::ReferenceException& e) {
        PyErr_SetString(h5referenceexception, e.getCDetailMsg());

    // Fallthrough to generic H5::Exception
    } catch (const H5::Exception& e) {
        PyErr_SetString(h5exception, e.getCDetailMsg());
    }
}
