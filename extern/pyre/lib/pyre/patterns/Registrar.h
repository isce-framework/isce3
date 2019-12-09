// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_patterns_Registrar_h)
#define pyre_patterns_Registrar_h

#include <map>
#include <string>

namespace pyre {
    namespace patterns {
        template <typename Object, typename Name> class Registrar;
    }
}

template <typename Object, typename Name = std::string>
class pyre::patterns::Registrar : public std::map<Name, Object *> {

    //typedefs
public:
    typedef Name name_t;
    typedef Object object_t;
    typedef std::map<name_t, object_t *> map_t;
    typedef typename map_t::iterator iterator_t;

    // interface
    inline ~Registrar();
};


// get the inline definitions
#define pyre_patterns_Registrar_icc
#include "Registrar.icc"
#undef pyre_patterns_Registrar_icc

#endif

// end of file
