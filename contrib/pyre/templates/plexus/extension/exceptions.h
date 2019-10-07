// -*- C++ -*-
// 
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
// 

#if !defined({project.name}_extension_exceptions_h)
#define {project.name}_extension_exceptions_h


// place everything in my private namespace
namespace {project.name} {{
    namespace extension {{

        // exception registration
        PyObject * registerExceptionHierarchy(PyObject *);

    }} // of namespace extension
}} // of namespace {project.name}

#endif

// end of file
