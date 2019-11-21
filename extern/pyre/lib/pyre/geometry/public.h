// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_geometry_public_h)
#define pyre_geometry_public_h

// externals
#include <stdexcept>
#include <array>
#include <valarray>
// support
#include <pyre/journal.h>
#include <pyre/memory.h>
#include <pyre/grid.h>

// forward declarations
namespace pyre {
    namespace geometry {
        // local type aliases
        typedef std::size_t size_t;
        // point
        template <size_t dim, typename dataT> class Point;
        // point cloud
        template <typename pointT> class PointCloud;
        // brick
        template <size_t dim, typename nodeT> class Brick;
    }
}

// type aliases for the above
namespace pyre {
    namespace geometry {
        // point
        template <size_t dim = 3, typename dataT = double> using point_t = Point<dim, dataT>;
        // point cloud
        template <typename pointT = point_t<3, double>> using cloud_t = PointCloud<pointT>;
        // brick
        template <size_t dim = 3, typename nodeT = point_t<3> > using brick_t = Brick<dim, nodeT>;
    }
}

// pull types from {pyre::grid}
namespace pyre {
    namespace geometry {
        template <typename repT>
        using index_t = pyre::grid::index_t<repT>;

        template <typename repT>
        using shape_t = pyre::grid::shape_t<repT>;

        template <size_t dim>
        using packing_t = pyre::grid::packing_t<dim>;

        template <typename indexT, typename packingT = packing_t<indexT::dim()>>
        using slice_t = pyre::grid::slice_t<indexT, packingT>;

        template <typename sliceT>
        using iterator_t = pyre::grid::iterator_t<sliceT>;

        template <typename indexT, typename packingT = packing_t<indexT::dim()>>
        using layout_t = pyre::grid::layout_t<indexT, packingT>;

        // grid
        template <typename cellT, typename layoutT, typename storageT>
        using grid_t = pyre::grid::grid_t<cellT, layoutT, storageT>;
        // direct grid
        template <typename cellT,
                  typename layoutT,
                  typename directT = pyre::memory::direct_t<cellT>>
        using directgrid_t = pyre::grid::directgrid_t<cellT, layoutT, directT>;
    }
}

// operators
namespace pyre {
    namespace geometry {
        // operators on points
        // equality
        template <size_t dim, typename dataT>
        inline
        auto operator== (const Point<dim, dataT> & p1, const Point<dim, dataT> & p2);
        // inequality
        template <size_t dim, typename dataT>
        inline
        auto operator!= (const Point<dim, dataT> & p1, const Point<dim, dataT> & p2);

        // operators on bricks
        // equality
        template <size_t dim, typename nodeT>
        inline
        auto operator== (const Brick<dim, nodeT> & b1, const Brick<dim, nodeT> & b2);
        // inequality
        template <size_t dim, typename nodeT>
        inline
        auto operator!= (const Brick<dim, nodeT> & b1, const Brick<dim, nodeT> & b2);

        // stream injection: overload the global operator<<
        // points
        template <size_t dim, typename dataT>
        inline
        auto & operator<< (std::ostream & stream, const Point<dim, dataT> & point);
        // bricks
        template <size_t dim, typename nodeT>
        inline
        auto & operator<< (std::ostream & stream, const Brick<dim, nodeT> & brick);
    }
}


// the object model
#include "Point.h"
#include "PointCloud.h"
#include "Brick.h"

// the implementations
// point
#define pyre_geometry_Point_icc
#include "Point.icc"
#undef pyre_geometry_Point_icc

// point cloud
#define pyre_geometry_PointCloud_icc
#include "PointCloud.icc"
#undef pyre_geometry_PointCloud_icc

// brick
#define pyre_geometry_Brick_icc
#include "Brick.icc"
#undef pyre_geometry_Brick_icc

# endif

// end of file
