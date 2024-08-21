#include "unwrap.h"
#include "ICU.h"
#include "Phass.h"

namespace py = pybind11;

void addsubmodule_unwrap(py::module & m)
{ 
    py::module m_unwrap = m.def_submodule("unwrap");
  
    // forward declare bound classes
    py::class_<isce3::unwrap::icu::ICU> pyICU(m_unwrap, "ICU");
    py::class_<isce3::unwrap::phass::Phass> pyPhass(m_unwrap, "Phass");    
  
    // add bindings
    addbinding(pyICU);
    addbinding(pyPhass);
}
