// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_journal_tests_h)
#define pyre_extensions_journal_tests_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace journal {

            // debug
            extern const char * const debugTest__name__;
            extern const char * const debugTest__doc__;
            PyObject * debugTest(PyObject *, PyObject *);
            // firewall
            extern const char * const firewallTest__name__;
            extern const char * const firewallTest__doc__;
            PyObject * firewallTest(PyObject *, PyObject *);
            // info
            extern const char * const infoTest__name__;
            extern const char * const infoTest__doc__;
            PyObject * infoTest(PyObject *, PyObject *);
            // warning
            extern const char * const warningTest__name__;
            extern const char * const warningTest__doc__;
            PyObject * warningTest(PyObject *, PyObject *);
            // error
            extern const char * const errorTest__name__;
            extern const char * const errorTest__doc__;
            PyObject * errorTest(PyObject *, PyObject *);

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
