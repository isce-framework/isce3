// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_algebra_operators_h)
#define pyre_algebra_operators_h


namespace pyre {
    namespace algebra {

        // binary operators
        template <typename numeric_t>
        inline
        numeric_t
        operator+ (const numeric_t &, const numeric_t &);

        template <typename numeric_t>
        inline
        numeric_t
        operator- (const numeric_t &, const numeric_t &);

        template <typename numeric_t>
        inline
        numeric_t
        operator* (const numeric_t &, const numeric_t &);

        template <typename numeric_t>
        inline
        numeric_t
        operator/ (const numeric_t &, const numeric_t &);
    }
}

// include the inlines
#include "operators.icc"
#endif


// end of file
