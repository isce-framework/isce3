// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// declaration of points
template <typename pointT>
class pyre::geometry::PointCloud {
    // types
public:
    typedef pointT point_type;
    typedef std::size_t size_type;
    typedef std::vector<point_type> rep_type;

    // meta-methods
public:
    inline explicit PointCloud(size_type count);

    // interface
public:
    // iteration
    inline auto begin() const;
    inline auto end() const;

    // indexed access
    inline auto & operator[](size_type pos);
    inline const auto & operator[](size_type pos) const;

    // implementation details
private:
    rep_type _points;
};


// end of file
