#include "TopoLayers.h"

#include <iterator>
#include <stdexcept>
#include <variant>
#include <vector>

namespace isce3::geometry {

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

} // namespace isce3::geometry
