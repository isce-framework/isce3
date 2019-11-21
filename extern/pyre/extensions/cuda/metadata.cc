// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

#include "metadata.h"

// the version number
#define CUDA_VERSION "1.0"

// copyright
PyObject *
pyre::extensions::cuda::
copyright(PyObject *, PyObject *)
{
    const char * const copyright_note = "cuda: (c) 1998-2019 orthologue";
    return Py_BuildValue("s", copyright_note);
}


// version
PyObject *
pyre::extensions::cuda::
version(PyObject *, PyObject *)
{
    const char * const version_string = "1.0";
    return Py_BuildValue("s", version_string);
}


// license
PyObject *
pyre::extensions::cuda::
license(PyObject *, PyObject *)
{
    const char * const license_string =
        "\n"
        "    cuda " CUDA_VERSION "\n"
        "    Copyright (c) 1998-2019 orthologue\n"
        "    All Rights Reserved\n"
        "\n"
        "    Redistribution and use in source and binary forms, with or without\n"
        "    modification, are permitted provided that the following conditions\n"
        "    are met:\n"
        "\n"
        "    * Redistributions of source code must retain the above copyright\n"
        "      notice, this list of conditions and the following disclaimer.\n"
        "\n"
        "    * Redistributions in binary form must reproduce the above copyright\n"
        "      notice, this list of conditions and the following disclaimer in\n"
        "      the documentation and/or other materials provided with the\n"
        "      distribution.\n"
        "\n"
        "    * Neither the name \"cuda\" nor the names of its contributors may be\n"
        "      used to endorse or promote products derived from this software\n"
        "      without specific prior written permission.\n"
        "\n"
        "    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"
        "    \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\n"
        "    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS\n"
        "    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE\n"
        "    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,\n"
        "    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,\n"
        "    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\n"
        "    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\n"
        "    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT\n"
        "    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN\n"
        "    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE\n"
        "    POSSIBILITY OF SUCH DAMAGE.\n";

    return Py_BuildValue("s", license_string);
}


// end of file
