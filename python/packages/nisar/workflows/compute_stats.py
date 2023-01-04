import isce3
from osgeo import gdal


def compute_stats_complex_data(raster, h5_ds):
    """
    Compute raster statistics for complex datasets

    raster: isce3.io.Raster
        ISCE3 Raster object
    h5_ds: h5py.File
        h5py file
    """
    stats_obj = isce3.math.compute_raster_stats_real_imag(raster)[0]
    write_stats_complex_data(h5_ds, stats_obj)


def write_stats_complex_data(h5_ds, stats_obj):
    h5_ds.attrs.create('min_real_value', data=stats_obj.real.min)
    h5_ds.attrs.create('mean_real_value', data=stats_obj.real.mean)
    h5_ds.attrs.create('max_real_value', data=stats_obj.real.max)
    h5_ds.attrs.create('sample_stddev_real',
                       data=stats_obj.real.sample_stddev)
    h5_ds.attrs.create('min_imag_value', data=stats_obj.imag.min)
    h5_ds.attrs.create('mean_imag_value', data=stats_obj.imag.mean)
    h5_ds.attrs.create('max_imag_value', data=stats_obj.imag.max)
    h5_ds.attrs.create('sample_stddev_imag',
                       data=stats_obj.imag.sample_stddev)


def compute_stats_real_data(raster, h5_ds):
    """
    Compute raster statistics for real datasets

    raster: isce3.io.Raster
       ISCE3 Raster object
    h5_ds: h5py.File
        h5py file
    """
    if raster.datatype() == gdal.GDT_Float64:
        stats_obj = isce3.math.compute_raster_stats_float64(raster)[0]
    else:
        stats_obj = isce3.math.compute_raster_stats_float32(raster)[0]
    h5_ds.attrs.create('min_value', data=stats_obj.min)
    h5_ds.attrs.create('mean_value', data=stats_obj.mean)
    h5_ds.attrs.create('max_value', data=stats_obj.max)
    h5_ds.attrs.create('sample_stddev',
                       data=stats_obj.sample_stddev)
