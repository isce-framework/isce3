import os

import iscetest
import pytest
from nisar.static.geo_grid import get_output_geo_grid

import isce3


class TestGetOutputGeoGrid:
    @pytest.fixture
    def dem_raster(self) -> isce3.io.Raster:
        dem_raster_file = os.path.join(iscetest.data, "dem_south_pole.tif")
        return isce3.io.Raster(dem_raster_file)

    def test_from_params(self, dem_raster: isce3.io.Raster):
        x_start = 0.0
        y_start = 2000.0
        x_spacing = 10.0
        y_spacing = 20.0
        width = 200
        length = 100
        epsg = 32701

        geo_grid = get_output_geo_grid(
            top_left={"x": x_start, "y": y_start},
            bottom_right={
                "x": (x_start + x_spacing * width),
                "y": (y_start - y_spacing * length),
            },
            posting={"x": x_spacing, "y": y_spacing},
            epsg=epsg,
            dem_raster=dem_raster,
        )

        assert geo_grid.start_x == x_start
        assert geo_grid.start_y == y_start
        assert geo_grid.spacing_x == x_spacing
        assert geo_grid.spacing_y == -y_spacing
        assert geo_grid.width == width
        assert geo_grid.length == length
        assert geo_grid.epsg == epsg

    def test_from_dem_raster(self, dem_raster: isce3.io.Raster):
        geo_grid = get_output_geo_grid(
            top_left={"x": None, "y": None},
            bottom_right={"x": None, "y": None},
            posting={"x": None, "y": None},
            epsg=None,
            dem_raster=dem_raster,
        )

        assert geo_grid.start_x == dem_raster.x0
        assert geo_grid.start_y == dem_raster.y0
        assert geo_grid.spacing_x == dem_raster.dx
        assert geo_grid.spacing_y == dem_raster.dy
        assert geo_grid.width == dem_raster.width
        assert geo_grid.length == dem_raster.length
        assert geo_grid.epsg == dem_raster.get_epsg()
