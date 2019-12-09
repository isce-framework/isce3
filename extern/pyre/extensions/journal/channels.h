// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_journal_channels_h)
#define pyre_extensions_journal_channels_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace journal {

            // lookup a name in the debug index
            extern const char * const lookupDebug__name__;
            extern const char * const lookupDebug__doc__;
            extern PyObject * lookupDebug(PyObject *, PyObject *);
            // lookup a name in the firewall index
            extern const char * const lookupFirewall__name__;
            extern const char * const lookupFirewall__doc__;
            extern PyObject * lookupFirewall(PyObject *, PyObject *);
            // lookup a name in the info index
            extern const char * const lookupInfo__name__;
            extern const char * const lookupInfo__doc__;
            extern PyObject * lookupInfo(PyObject *, PyObject *);
            // lookup a name in the warning index
            extern const char * const lookupWarning__name__;
            extern const char * const lookupWarning__doc__;
            extern PyObject * lookupWarning(PyObject *, PyObject *);
            // lookup a name in the error index
            extern const char * const lookupError__name__;
            extern const char * const lookupError__doc__;
            extern PyObject * lookupError(PyObject *, PyObject *);

            // access the state of Inventory<true>
            extern const char * const setEnabledState__name__;
            extern const char * const setEnabledState__doc__;
            extern PyObject * setEnabledState(PyObject *, PyObject *);

            extern const char * const getEnabledState__name__;
            extern const char * const getEnabledState__doc__;
            extern PyObject * getEnabledState(PyObject *, PyObject *);

            // access the state of Inventory<false>
            extern const char * const setDisabledState__name__;
            extern const char * const setDisabledState__doc__;
            extern PyObject * setDisabledState(PyObject *, PyObject *);

            extern const char * const getDisabledState__name__;
            extern const char * const getDisabledState__doc__;
            extern PyObject * getDisabledState(PyObject *, PyObject *);

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
