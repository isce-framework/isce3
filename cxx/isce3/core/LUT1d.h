#pragma once

#include "forward.h"
#include <isce3/core/EMatrix.h>
#include <isce3/except/Error.h>

#include <valarray>
#include <Eigen/Core>

namespace isce3::core {

/** Data structure to hold a 1D Lookup table.
 *  Suitable for storing data of the form y = f(x)*/
template <typename T>
class LUT1d {

    public:
        using ArrayXt = Eigen::Array<T, Eigen::Dynamic, 1>;
        /** Default constructor */
        inline LUT1d() : _haveData(false), _refValue(0.0), _extrapolate{true} {
            std::valarray<double> x{0.0, 1.0};
            std::valarray<double> y{0.0, 0.0};
            _coords = x;
            _values = y;
        }

        /** Constructor with a constant reference value */
        inline LUT1d(T refValue) : _haveData(false), _refValue(refValue), _extrapolate{true} {}

        /** Constructor with size */
        inline LUT1d(size_t size) : _haveData(false), _refValue(0.0), _extrapolate{true} {
            _coords.resize(size);
            _values.resize(size);
        }

        /** Constructor with coordinates and values
          * @param[in] coords Valarray for coordinates of LUT
          * @param[in] values Valarray for values of LUT
          * @param[in] extrapolate Flag for allowing extrapolation beyond bounds */
        inline LUT1d(const std::valarray<double> & coords, const std::valarray<T> & values,
                     bool extrapolate = false) :
                     _haveData(true),
                     _refValue(values[0]),
                     _coords(coords),
                     _values(values),
                     _extrapolate{extrapolate} {}

        /** Copy constructor.
          * @param[in] lut LUT1d object to copy from */
        inline LUT1d(const LUT1d<T> & lut) :
            _haveData(lut.haveData()),
            _refValue(lut.refValue()),
            _coords(lut.coords()),
            _values(lut.values()),
            _extrapolate(lut.extrapolate()) {}

        /** Assignment operator.
          * @param[in] lut LUT1d object to assign from */
        inline LUT1d & operator=(const LUT1d<T> & lut) {
            _haveData = lut.haveData();
            _refValue = lut.refValue();
            _coords = lut.coords();
            _values = lut.values();
            _extrapolate = lut.extrapolate();
            return *this;
        }

        /** Get a reference to the coordinates
          * @param[out] coords Reference to valarray for coordinates */
        inline std::valarray<double> & coords() { return _coords; }

        /** Get a read-only reference to the coordinates
          * @param[out] coords Copy of valarray for coordinates */
        inline const std::valarray<double> & coords() const { return _coords; }

        /** Set the coordinates
          * @param[in] c Input valarray for coordinates */
        inline void coords(const std::valarray<double> & c) { _coords = c; }

        /** Get a reference to the coordinates
          * @param[out] values Reference to valarray for values */
        inline std::valarray<T> & values() { return _values; }

        /** Get a read-only reference to the values
          * @param[out] values Copy of valarray for values */
        inline const std::valarray<T> & values() const { return _values; }

        /** Set the values
          * @param[in] v Input valarray for values */
        inline void values(const std::valarray<T> & v) { _values = v; }

        /** Get extrapolate flag
          * @param[out] flag Extrapolation flag */
        inline bool extrapolate() const { return _extrapolate; }

        /** Set extrapolate flag
          * @param[in] flag Extrapolation flag */
        inline void extrapolate(bool flag) { _extrapolate = flag; }

        /** Get flag for having data */
        inline bool haveData() const { return _haveData; }

        /** Get reference value */
        inline T refValue() const { return _refValue; }

        /** Get size of LUT
          * @param[out] size Size (number of coordinates) of LUT */
        inline size_t size() const { return _coords.size(); }

        /** Evaluate the LUT */
        inline T eval(double x) const;
        inline ArrayXt eval(const Eigen::Ref<const Eigen::ArrayXd>& x) const;
    // Data members
    private:
        bool _haveData;
        T _refValue;
        std::valarray<double> _coords;
        std::valarray<T> _values;
        bool _extrapolate;
};

/** Convert LUT2d to LUT1d by averaging along rows or columns
  * @param[out] lut1d LUT1d resulting from averaged LUT2d
  * @param[in] lut2d LUT2d to be converted
  * @param[in] axis Int Axis to be averaged. 0 for columns and 1 rows */
template <typename T>
LUT1d<T> avgLUT2dToLUT1d(const LUT2d<T> & lut2d, const int axis = 0);

} // namespace isce3::core

// Get inline implementations for LUT1d
#define ISCE_CORE_LUT1D_ICC
#include "LUT1d.icc"
#undef ISCE_CORE_LUT1D_ICC
