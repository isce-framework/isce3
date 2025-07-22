#include "DryTroposphereModel.h"

using isce3::focus::DryTroposphereModel;

namespace py = pybind11;

void addbinding(py::enum_<DryTroposphereModel> & pyDryTropoModel)
{
    pyDryTropoModel
        .value("NoDelay", DryTroposphereModel::NoDelay)
        .value("TSX", DryTroposphereModel::TSX);
}

void addbinding_tsx_delay(py::module& m)
{
    m.def(
      "dry_tropo_delay_tsx",
      &isce3::focus::dryTropoDelayTSX,
      py::arg("platform_xyz"),
      py::arg("target_llh"),
      py::arg("ellipsoid"),
      R"(
      Estimate dry tropospheric path delay using the TerraSAR-X model

      Parameters
      ----------
      platform_xyz : numpy.ndarray (float)
          Antenna phase center position in ECEF coordinate system in meters.
      target_llh : numpy.ndarray (float)
          Target position in LLH coordinates (longitude in radians, geodetic
          latitude in radians, height above ellipsoid in meters).
      ellipsoid : isce3.core.Ellipsoid
          The reference ellipsoid.
    
      Returns
      --------
      float
          The hydrostatic troposphere propagation delay, in seconds.
      
      References
      -----------
      .. [1] H. Breit, T. Fritz, U. Balss, M. Lachaise, A. Niedermeier and M.
          Vonavka, "TerraSAR-X SAR Processing and Products," in IEEE Transactions
          on Geoscience and Remote Sensing, vol. 48, no. 2, pp. 727-740, Feb.
          2010, doi: 10.1109/TGRS.2009.2035497.
      )"
    );
}
