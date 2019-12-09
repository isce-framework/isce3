// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_journal_macros_h)
#define pyre_journal_macros_h

//
// defined __HERE__, which has to be a preprocessor macro
//
#if defined(HAVE__FUNC__)

// gcc supports all three
#define __HERE__ __FILE__,__LINE__,__FUNCTION__
#define __HERE_ARGS__ filename, lineno, funcname
#define __HERE_DECL__ const char * filename, long lineno, const char * funcname

#else

#define __HERE__ __FILE__,__LINE__
#define __HERE_ARGS__ filename, lineno
#define __HERE_DECL__ const char * filename, long lineno

#endif // HAVE__FUNC__

#endif // pyre_journal_macros_h

// end of file
