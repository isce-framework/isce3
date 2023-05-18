#pragma once

#include "forward.h"

#include <string>
#include <valarray>

#include <isce3/io/Raster.h>

class isce3::geometry::TopoLayers {

    public:
        // Constructors
        TopoLayers(const std::string & outdir, const size_t length,
                const size_t width, const size_t linesPerBlock,
                const bool computeMask);

        TopoLayers(const size_t linesPerBlock,
                   isce3::io::Raster * xRaster = nullptr,
                   isce3::io::Raster * yRaster = nullptr,
                   isce3::io::Raster * zRaster = nullptr,
                   isce3::io::Raster * incRaster = nullptr,
                   isce3::io::Raster * hdgRaster = nullptr,
                   isce3::io::Raster * localIncRaster = nullptr,
                   isce3::io::Raster * localPsiRaster = nullptr,
                   isce3::io::Raster * simRaster = nullptr,
                   isce3::io::Raster * maskRaster = nullptr);

        // Destructor
        ~TopoLayers() {
            if (_haveOwnRasters) {
                delete _xRaster;
                delete _yRaster;
                delete _zRaster;
                delete _incRaster;
                delete _hdgRaster;
                delete _localIncRaster;
                delete _localPsiRaster;
                delete _simRaster;
                if (_maskRaster) {
                    delete _maskRaster;
                }
            }
        }

        // Set new block sizes
        void setBlockSize(size_t length, size_t width);

        // Get sizes
        inline size_t length() const { return _length; }
        inline size_t width() const { return _width; }

        // Get array references
        std::valarray<double> & x() { return _x; }
        std::valarray<double> & y() { return _y; }
        std::valarray<double> & z() { return _z; }
        std::valarray<float> & inc() { return _inc; }
        std::valarray<float> & hdg() { return _hdg; }
        std::valarray<float> & localInc() { return _localInc; }
        std::valarray<float> & localPsi() { return _localPsi; }
        std::valarray<float> & sim() { return _sim; }
        std::valarray<short> & mask() { return _mask; }
        std::valarray<double> & crossTrack() { return _crossTrack; }

        inline bool hasXRaster() const { return _xRaster != nullptr; }
        inline bool hasYRaster() const { return _yRaster != nullptr; }
        inline bool hasZRaster() const { return _zRaster != nullptr; }
        inline bool hasIncRaster() const { return _incRaster != nullptr; }
        inline bool hasHdgRaster() const { return _hdgRaster != nullptr; }
        inline bool hasLocalIncRaster() const { return _localIncRaster != nullptr; }
        inline bool hasLocalPsiRaster() const { return _localPsiRaster != nullptr; }
        inline bool hasSimRaster() const { return _simRaster != nullptr; }
        inline bool hasMaskRaster() const { return _maskRaster != nullptr; }

        // Set values for a single index
        inline void x(size_t row, size_t col, double value) {
            if (hasXRaster()) {
                _x[row*_width+col] = value;
            }
        }

        inline void y(size_t row, size_t col, double value) {
            if (hasYRaster()) {
                _y[row*_width + col] = value;
            }
        }

        inline void z(size_t row, size_t col, double value) {
            if (hasZRaster()) {
                _z[row*_width + col] = value;
            }
        }

        inline void inc(size_t row, size_t col, float value) {
            if (hasIncRaster()) {
                _inc[row*_width + col] = value;
            }
        }

        inline void hdg(size_t row, size_t col, float value) {
            if (hasHdgRaster()) {
                _hdg[row*_width + col] = value;
            }
        }

        inline void localInc(size_t row, size_t col, float value) {
            if (hasLocalIncRaster()) {
                _localInc[row*_width + col] = value;
            }
        }

        inline void localPsi(size_t row, size_t col, float value) {
            if (hasLocalPsiRaster()) {
                _localPsi[row*_width + col] = value;
            }
        }

        inline void sim(size_t row, size_t col, float value) {
            if (hasSimRaster()) {
             _sim[row*_width + col] = value;
            }
        }

        inline void mask(size_t row, size_t col, short value) {
            if (hasMaskRaster()) {
                _mask[row*_width + col] = value;
            }
        }

        inline void crossTrack(size_t row, size_t col, double value) {
            if (hasMaskRaster()) {
                _crossTrack[row*_width + col] = value;
            }
        }

        // Get values for a single index
        double x(size_t row, size_t col) const {
            if (!hasXRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return _x[row*_width+col];
        }

        double y(size_t row, size_t col) const {
            if (!hasYRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return _y[row*_width + col];
        }

        double z(size_t row, size_t col) const {
            if (!hasZRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return _z[row*_width + col];
        }

        float inc(size_t row, size_t col) const {
            if (!hasIncRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return _inc[row*_width + col];
        }

        float hdg(size_t row, size_t col) const {
            if (!hasHdgRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return _hdg[row*_width + col];
        }

        float localInc(size_t row, size_t col) const {
            if (!hasLocalIncRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return _localInc[row*_width + col];
        }

        float localPsi(size_t row, size_t col) const {
            if (!hasLocalPsiRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return _localPsi[row*_width + col];
        }

        float sim(size_t row, size_t col) const {
            if (!hasSimRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return _sim[row*_width + col];
        }

        short mask(size_t row, size_t col) const {
            if (!hasMaskRaster() || row > _length - 1 || col > _width -1) {
                return 0;
            }
            return _mask[row*_width + col];
        }

        double crossTrack(size_t row, size_t col) const {
            if (!hasMaskRaster() || row > _length - 1 || col > _width -1) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return _crossTrack[row*_width + col];
        }

        // Write data with rasters
        void writeData(size_t xidx, size_t yidx);

    private:
        // The valarrays for the actual data
        std::valarray<double> _x;
        std::valarray<double> _y;
        std::valarray<double> _z;
        std::valarray<float> _inc;
        std::valarray<float> _hdg;
        std::valarray<float> _localInc;
        std::valarray<float> _localPsi;
        std::valarray<float> _sim;
        std::valarray<short> _mask;
        std::valarray<double> _crossTrack; // internal usage only; not saved to Raster

        // Raster pointers for each layer
        isce3::io::Raster * _xRaster = nullptr;
        isce3::io::Raster * _yRaster = nullptr;
        isce3::io::Raster * _zRaster = nullptr;
        isce3::io::Raster * _incRaster = nullptr;
        isce3::io::Raster * _hdgRaster = nullptr;
        isce3::io::Raster * _localIncRaster = nullptr;
        isce3::io::Raster * _localPsiRaster = nullptr;
        isce3::io::Raster * _simRaster = nullptr;
        isce3::io::Raster * _maskRaster = nullptr;

        // Dimensions
        size_t _length, _width;

        // Directory for placing rasters
        std::string _topodir;

        // Flag indicates if the Class owns the Rasters
        // Should be True when initRasters is called
        // Should be false when the Rasters are passed
        // from outside and setRaster method is called
        bool _haveOwnRasters;
};
