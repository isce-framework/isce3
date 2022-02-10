#include "TopoLayers.h"

#include <iterator>
#include <stdexcept>
#include <variant>
#include <vector>
namespace isce3::geometry {

TopoLayers::TopoLayers(const std::string& outdir, const size_t length,
        const size_t width, const size_t linesPerBlock, const bool computeMask)
    : _length(length), _width(width), _haveOwnRasters(true)
{
    // Initialize the standard output rasters
    _xRaster = new isce3::io::Raster(
            outdir + "/x.rdr", width, length, 1, GDT_Float64, "ISCE");
    _yRaster = new isce3::io::Raster(
            outdir + "/y.rdr", width, length, 1, GDT_Float64, "ISCE");
    _zRaster = new isce3::io::Raster(
            outdir + "/z.rdr", width, length, 1, GDT_Float64, "ISCE");
    _incRaster = new isce3::io::Raster(
            outdir + "/inc.rdr", width, length, 1, GDT_Float32, "ISCE");
    _hdgRaster = new isce3::io::Raster(
            outdir + "/hdg.rdr", width, length, 1, GDT_Float32, "ISCE");
    _localIncRaster = new isce3::io::Raster(
            outdir + "/localInc.rdr", width, length, 1, GDT_Float32, "ISCE");
    _localPsiRaster = new isce3::io::Raster(
            outdir + "/localPsi.rdr", width, length, 1, GDT_Float32, "ISCE");
    _simRaster = new isce3::io::Raster(
            outdir + "/simamp.rdr", width, length, 1, GDT_Float32, "ISCE");

    // Optional mask raster
    if (computeMask) {
        _maskRaster = new isce3::io::Raster(
                outdir + "/layoverShadowMask.rdr", width, length, 1, GDT_Byte,
                "ISCE");
    } else {
        _maskRaster = nullptr;
    }

    // init valarrays to raster shape
    setBlockSize(linesPerBlock, width);
}

TopoLayers::TopoLayers(const size_t linesPerBlock, isce3::io::Raster* xRaster,
        isce3::io::Raster* yRaster, isce3::io::Raster* zRaster,
        isce3::io::Raster* incRaster, isce3::io::Raster* hdgRaster,
        isce3::io::Raster* localIncRaster, isce3::io::Raster* localPsiRaster,
        isce3::io::Raster* simRaster, isce3::io::Raster* maskRaster)
    : _haveOwnRasters(false)
{
    bool shape_set = false;

    // lambda to replace boilerplate of repeating this per raster
    auto setRaster = [&shape_set, this](isce3::io::Raster** dest,
                             isce3::io::Raster* src, const std::string& name) {
        // check if source raster pointer is not nullptr
        if (!src) {
            return;
        }
        if (shape_set) {
            // check if current source raster shape matches previous one
            if ((src->width() != _width) or (src->length() != _length)) {
                const std::string errmsg =
                        " raster shape is different from a preexisting "
                        "raster. All input rasters must have the same "
                        "shape.";
                throw isce3::except::LengthError(
                        ISCE_SRCINFO(), name + errmsg);
            }
        } else {
            // set shape parameters if not set
            _width = src->width();
            _length = src->length();
            shape_set = true;
        }

        // copy source raster to corresponding class raster
        *dest = src;
    };

    setRaster(&_xRaster, xRaster, "x");
    setRaster(&_yRaster, yRaster, "y");
    setRaster(&_zRaster, zRaster, "z");
    setRaster(&_incRaster, incRaster, "inc");
    setRaster(&_hdgRaster, hdgRaster, "hdg");
    setRaster(&_localIncRaster, localIncRaster, "localInc");
    setRaster(&_localPsiRaster, localPsiRaster, "localPsi");
    setRaster(&_simRaster, simRaster, "sim");
    setRaster(&_maskRaster, maskRaster, "layoverShadow");

    if (shape_set) {
        setBlockSize(linesPerBlock, _width);
    } else {
        const std::string errmsg = "No valid raster pointer passed in";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
}

void TopoLayers::writeData(size_t xidx, size_t yidx)
{
    std::vector<std::variant<double*, float*, short*>> valarrays {&_x[0],
            &_y[0], &_z[0], &_inc[0], &_hdg[0], &_localInc[0], &_localPsi[0],
            &_sim[0], &_mask[0]};

    std::vector<isce3::io::Raster*> rasters {_xRaster, _yRaster, _zRaster,
            _incRaster, _hdgRaster, _localIncRaster, _localPsiRaster,
            _simRaster, _maskRaster};

#pragma omp parallel for
    for (auto i = 0; i < valarrays.size(); ++i) {
        if (rasters[i]) {
            // std::bad_variant_access requires macOS 10.14
            if (auto* p = std::get_if<double*>(&valarrays[i])) {
                rasters[i]->setBlock(*p, xidx, yidx, _width, _length);
            } else if (auto* p = std::get_if<float*>(&valarrays[i])) {
                rasters[i]->setBlock(*p, xidx, yidx, _width, _length);
            } else if (auto* p = std::get_if<short*>(&valarrays[i])) {
                rasters[i]->setBlock(*p, xidx, yidx, _width, _length);
            } else {
                throw std::logic_error("invalid variant type");
            }
        }
    }
}

// Set new block sizes
void TopoLayers::setBlockSize(size_t length, size_t width)
{
    _length = length;
    _width = width;
    _x.resize(length * width);
    _y.resize(length * width);
    _z.resize(length * width);
    _inc.resize(length * width);
    _hdg.resize(length * width);
    _localInc.resize(length * width);
    _localPsi.resize(length * width);
    _sim.resize(length * width);
    _mask.resize(length * width);
    _crossTrack.resize(length * width);
}

// Check if only x, y, and z rasters are enabled
bool TopoLayers::onlyXYZRastersSet() const
{
    // Check if x, y, and z rasters are all not nullptr
    bool has_xyz = _xRaster && _yRaster && _zRaster;

    // Check if any other rasters not nullptr
    bool nothing_else = !_incRaster && !_hdgRaster && !_localIncRaster
        && !_localPsiRaster && !_simRaster && !_maskRaster;

    return has_xyz && nothing_else;
}

} // namespace isce3::geometry
