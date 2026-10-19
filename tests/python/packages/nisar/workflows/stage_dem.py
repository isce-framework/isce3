import os
import subprocess
import argparse
from glob import glob
from tempfile import TemporaryDirectory

import numpy.testing as npt
import pytest
from nisar.workflows import stage_dem
from osgeo import gdal, osr

from shapely import wkt

import iscetest


def test_dateline_crossing():
    # Polygon on Ross Ice Shelf (Antarctica)
    # crossing the dateline
    polygon_wkt = 'POLYGON((-160.9795 -76.9215,163.3981 -77.0962,' \
                  '          152.885 -81.8908,-149.3722 -81.6129,-160.9795 -76.9215))'
    polygon = wkt.loads(polygon_wkt)

    # Check if crossing dateline
    polys = stage_dem.check_dateline(polygon)

    assert (len(polys) == 2)


@pytest.mark.parametrize("area_or_point", ["Area", "Point"])
def test_translate_dem_dateline_shift_keeps_crs(area_or_point: str):
    # A crop starting at -180 deg is shifted by +360 deg so that it
    # mosaics with the tile on the other side of the dateline. The
    # shifted GeoTIFF must keep its CRS, including for DEMs tagged
    # AREA_OR_POINT=Point, or gdal.BuildVRT drops it from the mosaic.
    with TemporaryDirectory() as tmpdir:
        # Synthetic 1 arcsec EPSG:4326 DEM whose left edge sits half
        # a pixel west of -180 deg, like the NISAR DEM
        spacing = 1.0 / 3600.0
        x0 = -180.0 - spacing / 2
        y0 = -16.0 + spacing / 2
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        src = os.path.join(tmpdir, "dem.tif")
        ds = gdal.GetDriverByName("GTiff").Create(src, 720, 720, 1,
                                                  gdal.GDT_Float32)
        ds.SetMetadataItem("AREA_OR_POINT", area_or_point)
        ds.SetGeoTransform((x0, spacing, 0.0, y0, 0.0, -spacing))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).Fill(100.0)
        ds = None

        outpath = os.path.join(tmpdir, "dem_0.tiff")
        stage_dem.translate_dem(src, outpath, -180.0, -179.9, -16.1, -16.0,
                                4326)

        ds = gdal.Open(outpath, gdal.GA_ReadOnly)
        out_srs = osr.SpatialReference(wkt=ds.GetProjection())
        assert out_srs.GetAuthorityCode(None) == "4326"
        npt.assert_allclose(ds.GetGeoTransform()[0], x0 + 360.0)


def test_point2epsg():
    # Coordinate points taken from track-frame database

    npt.assert_equal(stage_dem.point2epsg(-162.377, 0.881), 32603)
    npt.assert_equal(stage_dem.point2epsg(-108.511, 8.901), 32612)
    npt.assert_equal(stage_dem.point2epsg(151.555, -34.963), 32756)
    npt.assert_equal(stage_dem.point2epsg(81.863, 0.817), 32644)
    npt.assert_equal(stage_dem.point2epsg(-21.748, 80.49), 3413)
    npt.assert_equal(stage_dem.point2epsg(77.191, -79.856), 3031)


def test_run_stage_dem():
    # Argparse
    opts = argparse.Namespace(
        product=os.path.join(iscetest.data, "Greenland.h5"),
        outfile='dem.vrt', bbox=None, filepath='file',
        margin=1, version='1.1')

    try:
        stage_dem.main(opts)
    except subprocess.CalledProcessError:
        print('S3 bucket not accessible')


def test_check_overlap():
    # Determine poliygon covered by RSLC
    rslc_file = os.path.join(iscetest.data, 'Greenland.h5')

    poly = stage_dem.determine_polygon(rslc_file, None)
    polys = stage_dem.check_dateline(poly)

    # Determine overlap
    overlap = stage_dem.check_dem_overlap('dem.vrt', polys)

    npt.assert_allclose(overlap, 100, rtol=1e-3)


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2"])
def test_dem_description(version: str):
    with TemporaryDirectory() as tmpdir:
        product = os.path.join(iscetest.data, "Greenland.h5")
        outfile = os.path.join(tmpdir, "dem.vrt")
        opts = argparse.Namespace(
            product=product,
            outfile=outfile,
            bbox=None,
            filepath="file",
            margin=1,
            version=version,
        )
        stage_dem.main(opts)

        ds = gdal.Open(outfile, gdal.GA_ReadOnly)
        vrt_descr = ds.GetMetadataItem("dem_description")
        assert len(vrt_descr) > 0

        tiff_files = glob(os.path.join(tmpdir, "dem_*.tiff"))
        assert len(tiff_files) > 0

        for tiff_file in tiff_files:
            ds = gdal.Open(tiff_file, gdal.GA_ReadOnly)
            tiff_descr = ds.GetMetadataItem("dem_description")
            assert tiff_descr == vrt_descr


if __name__ == "__main__":
    test_dateline_crossing()
    test_point2epsg()
    test_run_stage_dem()
    test_check_overlap()
