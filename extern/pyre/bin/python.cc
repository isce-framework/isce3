// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// this a slightly augmented python interpreter
// it contains Py_break, a function that enables the debugging of extensions

// externals
#include <Python.h>
#include <string>
#include <locale.h>

// declarations
void Py_break(int);
static inline wchar_t * _widen(const char *);

// the python main entry point
int main(int argc, char *argv[]) {
    // get the current locale
    std::string savedLocale = setlocale(LC_ALL, 0);
    // adjust it
    setlocale(LC_ALL, "");

    // the interpreter main entry point wants command line arguments as wchar_t; so we have to
    // convert
    wchar_t **wargv = new wchar_t* [argc+1];
    // go through the command line arguments
    for (int arg=0; arg < argc; ++arg) {
        // convert and store each one
        wargv[arg] = _widen(argv[arg]);
    }
    // place the sentinel
    wargv[argc] = 0;

    // invoke the dummy breakpoint function
    Py_break(0);

    // invoke the interpreter
    int status = Py_Main(argc, wargv);

    // restore the old locale
    setlocale(LC_ALL, savedLocale.c_str());
    // clean up
    delete [] wargv;
    // all done
    return status;
}

// helpers
// a global variable to be "modified" by the dummy routine so we can fool the optimizer
int Py_breakpoint = -1;
// the dummy routine
void Py_break(int value) {
    // set the global
    Py_breakpoint = value;
    // all done
    return;
}

// the version dependent conversion of char command line arguments to wchar
wchar_t * _widen(const char * arg) {
    // in python 3.5+
#if PY_VERSION_HEX >= 0x03050000
    return  Py_DecodeLocale(arg, 0);
    // up to python 3.4
#else
     return  _Py_char2wchar(arg, 0);
#endif
}

// end of file
