// -*- C++ -*-
// 
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
// 

#if !defined({project.name}_extension_metadata_h)
#define {project.name}_extension_metadata_h


// place everything in my private namespace
namespace {project.name} {{
    namespace extension {{

        // copyright note
        extern const char * const copyright__name__;
        extern const char * const copyright__doc__;
        PyObject * copyright(PyObject *, PyObject *);

        // version
        extern const char * const version__name__;
        extern const char * const version__doc__;
        PyObject * version(PyObject *, PyObject *);

    }} // of namespace extension`
}} // of namespace {project.name}

#endif

// end of file
