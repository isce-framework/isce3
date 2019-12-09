/* -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
*/

#if !defined(pyre_journal_debuginfo_h)
#define pyre_journal_debuginfo_h

/* the __HERE__ macros */
#include "macros.h"

/* build the declarations of the bindings in a C-compatible way */
#ifdef __cplusplus
extern "C" {
#endif

    int debuginfo_active(const char * channel);
    void debuginfo_activate(const char * channel);
    void debuginfo_deactivate(const char * channel);
    void debuginfo_out(const char * channel, __HERE_DECL__, const char * fmt, ...);

#ifdef __cplusplus
}
#endif

# endif // pyre_journal_debuginfo_h

/* end of file */
