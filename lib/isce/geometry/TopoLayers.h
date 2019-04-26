//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_TOPOLAYERS_H
#define ISCE_GEOMETRY_TOPOLAYERS_H

#include <valarray>
#include <string>
#include <isce/io/Raster.h>

// Declaration
namespace isce {
    namespace geometry {
        class TopoLayers;
    }
}

class isce::geometry::TopoLayers {

    public:
        // Default constructor
        TopoLayers() : _length(0.0), _width(0.0), _haveRasters(false) {}
        // Constructors
        TopoLayers(size_t length, size_t width) : _length(length), _width(width),
                                                  _haveRasters(false) {
            _x.resize(length*width);
            _y.resize(length*width);
            _z.resize(length*width);
            _inc.resize(length*width);
            _hdg.resize(length*width);
            _localInc.resize(length*width);
            _localPsi.resize(length*width);
            _sim.resize(length*width);
            _mask.resize(length*width);
            _crossTrack.resize(length*width);
        }
        // Destructor
        ~TopoLayers() {
            if (_haveRasters) {
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
        void setBlockSize(size_t length, size_t width) {
            _length = length;
            _width = width;
            _x.resize(length*width);
            _y.resize(length*width);
            _z.resize(length*width);
            _inc.resize(length*width);
            _hdg.resize(length*width);
            _localInc.resize(length*width);
            _localPsi.resize(length*width);
            _sim.resize(length*width);
            _mask.resize(length*width);
            _crossTrack.resize(length*width);
        }

        // Get sizes
        inline size_t length() const { return _length; }
        inline size_t width() const { return _width; }

        // Initialize rasters
        void initRasters(const std::string & outdir, size_t width, size_t length,
                         bool computeMask = false) {

            // Initialize the standard output rasters
            _xRaster = new isce::io::Raster(outdir + "/x.rdr", width, length, 1,
                GDT_Float64, "ISCE");
            _yRaster = new isce::io::Raster(outdir + "/y.rdr", width, length, 1,
                GDT_Float64, "ISCE");
            _zRaster = new isce::io::Raster(outdir + "/z.rdr", width, length, 1,
                GDT_Float64, "ISCE");
            _incRaster = new isce::io::Raster(outdir + "/inc.rdr", width, length, 1,
                GDT_Float32, "ISCE");
            _hdgRaster = new isce::io::Raster(outdir + "/hdg.rdr", width, length, 1,
                GDT_Float32, "ISCE");
            _localIncRaster = new isce::io::Raster(outdir + "/localInc.rdr", width, length, 1,
                GDT_Float32, "ISCE");
            _localPsiRaster = new isce::io::Raster(outdir + "/localPsi.rdr", width, length, 1,
                GDT_Float32, "ISCE");
            _simRaster = new isce::io::Raster(outdir + "/simamp.rdr", width, length, 1,
                GDT_Float32, "ISCE");
       
            // Optional mask raster
            if (computeMask) { 
                _maskRaster = new isce::io::Raster(outdir + "/mask.rdr", width, length, 1,
                    GDT_Byte, "ISCE");
            } else {
                _maskRaster = nullptr;
            }

            // Update sizes
            _width = width;
            _length = length;

            // Indicate that we have initialized rasters
            _haveRasters = true;
        }

        // Set rasters (plus mask raster) from externally created rasters
        void setRasters(isce::io::Raster & xRaster, isce::io::Raster & yRaster,
                        isce::io::Raster & zRaster, isce::io::Raster & incRaster,
                        isce::io::Raster & hdgRaster, isce::io::Raster & localIncRaster,
                        isce::io::Raster & localPsiRaster, isce::io::Raster & simRaster) {
            _xRaster = &xRaster;
            _yRaster = &yRaster;
            _zRaster = &zRaster;
            _incRaster = &incRaster;
            _hdgRaster = &hdgRaster;
            _localIncRaster = &localIncRaster;
            _localPsiRaster = &localPsiRaster;
            _simRaster = &simRaster;
        }

        // Set rasters (plus mask raster) from externally created rasters
        void setRasters(isce::io::Raster & xRaster, isce::io::Raster & yRaster,
                        isce::io::Raster & zRaster, isce::io::Raster & incRaster,
                        isce::io::Raster & hdgRaster, isce::io::Raster & localIncRaster,
                        isce::io::Raster & localPsiRaster, isce::io::Raster & simRaster,
                        isce::io::Raster & maskRaster) {
            _xRaster = &xRaster;
            _yRaster = &yRaster;
            _zRaster = &zRaster;
            _incRaster = &incRaster;
            _hdgRaster = &hdgRaster;
            _localIncRaster = &localIncRaster;
            _localPsiRaster = &localPsiRaster;
            _simRaster = &simRaster;
            _maskRaster = &maskRaster;
        }

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
        
        // Set values for a single index
        void x(size_t row, size_t col, double value) {
            _x[row*_width+col] = value;
        }
        
        void y(size_t row, size_t col, double value) {
            _y[row*_width + col] = value;
        }
        
        void z(size_t row, size_t col, double value) {
            _z[row*_width + col] = value;
        }
        
        void inc(size_t row, size_t col, float value) {
            _inc[row*_width + col] = value;
        }
        
        void hdg(size_t row, size_t col, float value) {
            _hdg[row*_width + col] = value;
        }
        
        void localInc(size_t row, size_t col, float value) {
            _localInc[row*_width + col] = value;
        }
        
        void localPsi(size_t row, size_t col, float value) {
            _localPsi[row*_width + col] = value;
        }
    
        void sim(size_t row, size_t col, float value) {
            _sim[row*_width + col] = value;
        }

        void mask(size_t row, size_t col, short value) {
            _mask[row*_width + col] = value;
        }

        void crossTrack(size_t row, size_t col, double value) {
            _crossTrack[row*_width + col] = value;
        }

        // Get values for a single index
        double x(size_t row, size_t col) const {
            return _x[row*_width+col];
        }
        
        double y(size_t row, size_t col) const {
            return _y[row*_width + col];
        }
        
        double z(size_t row, size_t col) const {
            return _z[row*_width + col];
        }
        
        float inc(size_t row, size_t col) const {
            return _inc[row*_width + col];
        }
        
        float hdg(size_t row, size_t col) const {
            return _hdg[row*_width + col];
        }
        
        float localInc(size_t row, size_t col) const {
            return _localInc[row*_width + col];
        }
        
        float localPsi(size_t row, size_t col) const {
            return _localPsi[row*_width + col];
        }
    
        float sim(size_t row, size_t col) const {
            return _sim[row*_width + col];
        }

        short mask(size_t row, size_t col) const {
            return _mask[row*_width + col];
        }

        double crossTrack(size_t row, size_t col) const {
            return _crossTrack[row*_width + col];
        }

        // Write data with rasters
        void writeData(size_t xidx, size_t yidx) {
            _xRaster->setBlock(_x, xidx, yidx, _width, _length);
            _yRaster->setBlock(_y, xidx, yidx, _width, _length);
            _zRaster->setBlock(_z, xidx, yidx, _width, _length);
            _incRaster->setBlock(_inc, xidx, yidx, _width, _length);
            _hdgRaster->setBlock(_hdg, xidx, yidx, _width, _length);
            _localIncRaster->setBlock(_localInc, xidx, yidx, _width, _length);
            _localPsiRaster->setBlock(_localPsi, xidx, yidx, _width, _length);
            _simRaster->setBlock(_sim, xidx, yidx, _width, _length);
            if (_maskRaster) {
                _maskRaster->setBlock(_mask, xidx, yidx, _width, _length);
            }
        }
        
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
        isce::io::Raster * _xRaster;
        isce::io::Raster * _yRaster;
        isce::io::Raster * _zRaster;
        isce::io::Raster * _incRaster;
        isce::io::Raster * _hdgRaster;
        isce::io::Raster * _localIncRaster;
        isce::io::Raster * _localPsiRaster;
        isce::io::Raster * _simRaster;
        isce::io::Raster * _maskRaster;

        // Dimensions
        size_t _length, _width;

        // Directory for placing rasters
        std::string _topodir;
        bool _haveRasters;
};
    
#endif

// end of file
