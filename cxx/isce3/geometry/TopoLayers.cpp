#include "TopoLayers.h"

#include <iterator>
#include <variant>
#include <vector>

namespace isce3::geometry {

void TopoLayers::writeData(size_t xidx, size_t yidx)
{
    std::vector<std::variant<double *, float *, short *>>
        valarrays{&_x[0], &_y[0], &_z[0], &_inc[0], &_hdg[0], &_localInc[0],
                  &_localPsi[0], &_sim[0], &_mask[0]};

    std::vector<isce3::io::Raster *> rasters{_xRaster, _yRaster, _zRaster,
        _incRaster, _hdgRaster, _localIncRaster, _localPsiRaster,
        _simRaster, _maskRaster};

    #pragma omp parallel for
    for (auto i = 0; i < valarrays.size(); ++i)
    {
        if (rasters[i])
        {
            std::visit([&](const auto& ptr) {
                rasters[i]->setBlock(ptr, xidx, yidx, _width, _length);
            }, valarrays[i]);
        }
    }
}

}
