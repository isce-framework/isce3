#!/usr/bin/env python3

# WATERMASK staging

import argparse
import os
import backoff

import numpy as np
import shapely.ops
import shapely.wkt
from osgeo import gdal, osr
from shapely.geometry import LinearRing, Point, Polygon, box


# Enable exceptions
gdal.UseExceptions()

EARTH_APPROX_CIRCUMFERENCE = 40075017.
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)

STATIC_REPO = 'nisar-static-repo'
WATER_MASK_VSIS3_PATH = f'/vsis3/{STATIC_REPO}/WATER_MASK'


def cmdLineParse():
    """
     Command line parser
    """
    parser = argparse.ArgumentParser(description="""
                                     Stage and verify WATERMASK for processing. """,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--product', type=str, action='store',
                        help='Input reference RSLC HDF5 product')
    parser.add_argument('-o', '--output', type=str, action='store',
                        default='WATERMASK.vrt', dest='outfile',
                        help='Output water mask filepath (VRT format).')
    parser.add_argument('-f', '--path', type=str, action='store',
                        dest='filepath', default='file',
                        help='Filepath to user water mask.')
    parser.add_argument('-m', '--margin', type=int, action='store',
                        default=5, help='Margin for water mask bounding box (km)')
    parser.add_argument('-v', '--version', type=str, action='store',
                        dest='version', default='0.3',
                        help='Version for water mask')
    parser.add_argument('-b', '--bbox', type=float, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in latitude/longitude (WSEN, decimal degrees)')
    return parser.parse_args()


def check_dateline(poly):
    """Split `poly` if it crosses the dateline.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    polys : list of shapely.geometry.Polygon
         A list containing: the input polygon if it didn't cross
        the dateline, or two polygons otherwise (one on either
        side of the dateline).
    """

    xmin, _, xmax, _ = poly.bounds
    # Check dateline crossing
    if (xmax - xmin) > 180.0:
        dateline = shapely.wkt.loads('LINESTRING( 180.0 -90.0, 180.0 90.0)')

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        # The Copernicus DEM used for NISAR processing has a longitude
        # range [-180, +180]. The current version of gdal.Translate
        # does not allow to perform dateline wrapping. Therefore, coordinates
        # above 180 need to be wrapped down to -180 to match the Copernicus
        # DEM longitude range
        for polygon_count in range(2):
            x, y = polys[polygon_count].exterior.coords.xy
            if not any(k > 180 for k in x):
                continue

            # Otherwise, wrap longitude values down to 360 deg
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

        assert len(polys) == 2
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def determine_polygon(ref_slc, bbox=None):
    """Determine bounding polygon using RSLC radar grid/orbit
    or user-defined bounding box

    Parameters:
    ----------
    ref_slc: str
        Filepath to reference RSLC product
    bbox: list, float
        Bounding box with lat/lon coordinates (decimal degrees)
        in the form of [West, South, East, North]

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RSLC perimeter
        or bbox shape on the ground
    """
    if bbox is not None:
        print('Determine polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        print('Determine polygon from RSLC radar grid and orbit')
        poly = get_geo_polygon(ref_slc)

    return poly


def point2epsg(lon, lat):
    """Return EPSG code based on point lat/lon

    Parameters:
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point

    Returns:
    -------
    epsg code corresponding to the point lat/lon coordinates
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 75.0:
        return 3413
    elif lat <= -75.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    else:
        raise ValueError(
            'Could not determine projection for {0},{1}'.format(lat, lon))


def get_geo_polygon(ref_slc, min_height=-500.,
                    max_height=9000., pts_per_edge=5):
    """Create polygon (EPSG:4326) using RSLC radar grid and orbits

    Parameters:
    -----------
    ref_slc: str
        Path to RSLC product to stage the water mask for
    min_height: float
        Global minimum height (in m) for water mask interpolator
    max_height: float
        Global maximum height (in m) for water mask interpolator
    pts_per_edge: float
        Number of points per edge for min/max bounding box computation

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RSLC perimeter on the ground
    """
    from isce3.core import LUT2d  # pylint: disable=import-error
    from isce3.geometry import DEMInterpolator, get_geo_perimeter_wkt  # pylint: disable=import-error
    from nisar.products.readers import SLC  # pylint: disable=import-error

    # Prepare SLC dataset input
    productSlc = SLC(hdf5file=ref_slc)

    # Extract orbits, radar grid, and doppler for frequency A
    orbit = productSlc.getOrbit()
    radar_grid = productSlc.getRadarGrid(frequency='A')
    doppler = LUT2d()

    # Get min and max global height WATERMASK interpolators
    watermask_min = DEMInterpolator(height=min_height)
    watermask_max = DEMInterpolator(height=max_height)

    # Get min and max bounding boxes
    box_min = get_geo_perimeter_wkt(radar_grid, orbit, doppler,
                                    watermask_min, pts_per_edge)
    box_max = get_geo_perimeter_wkt(radar_grid, orbit, doppler,
                                    watermask_max, pts_per_edge)

    # Determine minimum and maximum polygons
    poly_min = shapely.wkt.loads(box_min)
    poly_max = shapely.wkt.loads(box_max)

    # Get polygon from intersection of poly_min and poly_max
    poly = poly_min | poly_max

    return poly


def determine_projection(polys):
    """Determine EPSG code for each polygon in polys.
    EPSG is computed for a regular list of points. EPSG
    is assigned based on a majority criteria.

    Parameters:
    -----------
    polys: shapely.Geometry.Polygon
        List of shapely Polygons

    Returns:
    --------
    epsg:
        List of EPSG codes corresponding to elements in polys
    """

    epsg = []

    # Make a regular grid based on polys min/max latitude longitude
    for p in polys:
        xmin, ymin, xmax, ymax = p.bounds
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250),
                             np.linspace(ymin, ymax, 250))
        x = xx.flatten()
        y = yy.flatten()

        # Query to determine the zone
        zones = []
        for lx, ly in zip(x, y):
            # Create a point with grid coordinates
            pp = Point(lx, ly)
            # If Point is in polys, compute EPSG
            if pp.within(p):
                zones.append(point2epsg(lx, ly))

        # Count different EPSGs
        vals, counts = np.unique(zones, return_counts=True)
        # Get the ESPG for Polys
        epsg.append(vals[np.argmax(counts)])

    return epsg


@backoff.on_exception(backoff.expo, Exception, max_tries=8, max_value=32)
def translate_watermask(vrt_filename, outpath, x_min, x_max, y_min, y_max):
    """Translate water mask from nisar-WATERMASK bucket. This
       function is decorated to perform retries
       using exponential backoff to make the remote
       call resilient to transient issues stemming
       from network access, authorization and AWS
       throttling (see "Query throttling" section at
       https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html).

    Parameters:
    ----------
    vrt_filename: str
        Path to the input VRT file
    outpath: str
        Path to the translated output GTiff file
    x_min: float
        Minimum longitude bound of the subwindow
    x_max: float
        Maximum longitude bound of the subwindow
    y_min: float
        Minimum latitude bound of the subwindow
    y_max: float
        Maximum latitude bound of the subwindow
    """

    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    input_x_min, xres, _, input_y_max, _, yres = ds.GetGeoTransform()
    length = ds.GetRasterBand(1).YSize
    width = ds.GetRasterBand(1).XSize
    input_y_min = input_y_max + (length * yres)
    input_x_max = input_x_min + (width * xres)

    x_min = max(x_min, input_x_min)
    x_max = min(x_max, input_x_max)
    y_min = max(y_min, input_y_min)
    y_max = min(y_max, input_y_max)

    gdal.Translate(outpath, ds, format='GTiff',
                   projWin=[x_min, y_max, x_max, y_min])
    ds = None


def download_watermask(polys, epsgs, outfile, version):
    """Download water mask from nisar-WATERMASK bucket

    Parameters:
    ----------
    polys: shapely.geometry.Polygon
        List of shapely polygons
    epsg: str, list
        List of EPSG codes corresponding to polys
    outfile:
        Path to the output WATERMASK file to be staged
    version: str
        Water mask version
    """
    try:
        # Version 0.2 does have geographic coordinate (EPSG4326),
        # without polar stereographic coordinates
        if version != '0.2':
            if 3031 in epsgs:
                epsgs = [3031] * len(epsgs)
                polys = transform_polygon_coords(polys, epsgs)
                # Need one EPSG as in polar stereo; we have one big polygon
                epsgs = [3031]
            elif 3413 in epsgs:
                epsgs = [3413] * len(epsgs)
                polys = transform_polygon_coords(polys, epsgs)
                # Need one EPSG as in polar stereo; we have one big polygon
                epsgs = [3413]
            else:
                # set epsg to 4326 for each element in the list
                epsgs = [4326] * len(epsgs)

        # Download WATERMASK for each polygon/epsg
        file_prefix = os.path.splitext(outfile)[0]
        watermask_list = []
        for n, (epsg, poly) in enumerate(zip(epsgs, polys)):
            if version == '0.2':
                vrt_filename = f'{WATER_MASK_VSIS3_PATH}/v{version}/watermask.vrt'
            else:
                vrt_filename = f'{WATER_MASK_VSIS3_PATH}/v{version}/EPSG{epsg}.vrt'
            outpath = f'{file_prefix}_{n}.tiff'
            watermask_list.append(outpath)
            xmin, ymin, xmax, ymax = poly.bounds
            translate_watermask(vrt_filename, outpath, xmin, xmax, ymin, ymax)

        # Build vrt with downloaded watermasks
        gdal.BuildVRT(outfile, watermask_list)
    except Exception:
        errmsg = f'Failed to donwload NISAR WATERMASK {version} from s3 bucket. ' \
                 f'Maybe {version} is not currently supported.'
        raise ValueError(errmsg)


def transform_polygon_coords(polys, epsgs):
    """Transform coordinates of polys (list of polygons)
       to target epsgs (list of EPSG codes)

    Parameters:
    ----------
    polys: shapely.Geometry.Polygon
        List of shapely polygons
    epsg: list, str
        List of EPSG codes corresponding to
        elements in polys
    """

    # Assert validity of inputs
    assert len(polys) == len(epsgs)

    # Transform each point of the perimeter in target EPSG coordinates
    llh = osr.SpatialReference()
    llh.ImportFromEPSG(4326)
    tgt = osr.SpatialReference()

    xmin, ymin, xmax, ymax = [], [], [], []
    tgt_x, tgt_y = [], []
    for poly, epsg in zip(polys, epsgs):
        x, y = poly.exterior.coords.xy
        tgt.ImportFromEPSG(int(epsg))
        trans = osr.CoordinateTransformation(llh, tgt)
        for lx, ly in zip(x, y):
            dummy_x, dummy_y, dummy_z = trans.TransformPoint(ly, lx, 0)
            tgt_x.append(dummy_x)
            tgt_y.append(dummy_y)
        xmin.append(min(tgt_x))
        ymin.append(min(tgt_y))
        xmax.append(max(tgt_x))
        ymax.append(max(tgt_y))
    # return a polygon
    poly = [Polygon([(min(xmin), min(ymin)), (min(xmin), max(ymax)),
                     (max(xmax), max(ymax)), (max(xmax), min(ymin))])]

    return poly


def check_watermask_overlap(watermaskFilepath, polys):
    """Evaluate overlap between user-provided WATERMASK
       and WATERMASK that stage_watermask.py would download
       based on RSLC or bbox provided information

    Parameters:
    ----------
    watermaskFilepath: str
        Filepath to the user-provided WATERMASK
    polys: shapely.geometry.Polygon
        List of polygons computed from RSLC or bbox

    Returns:
    -------
    perc_area: float
        Area (in percentage) covered by the intersection between the
        user-provided WATERMASK and the one downloadable by stage_watermask.py
    """
    from isce3.io import Raster  # pylint: disable=import-error

    # Get local WATERMASK edge coordinates
    watermask = Raster(watermaskFilepath)
    ulx, xres, xskew, uly, yskew, yres = watermask.get_geotransform()
    lrx = ulx + (watermask.width * xres)
    lry = uly + (watermask.length * yres)
    poly_watermask = Polygon([(ulx, uly), (ulx, lry), (lrx, lry), (lrx, uly)])

    # Initialize epsg
    epsg = [watermask.get_epsg()] * len(polys)

    if watermask.get_epsg() != 4326:
        polys = transform_polygon_coords(polys, epsg)

    perc_area = 0
    for poly in polys:
        perc_area += (poly.intersection(poly_watermask).area / poly.area) * 100
    return perc_area


def check_aws_connection():
    """Check connection to AWS s3://nisar-static-repo/WATER_MASK bucket
       Throw exception if no connection is established

    Parameters
    ----------
    version: str
        Version for water mask
    """
    import boto3
    s3 = boto3.resource('s3')
    # Check only AWS connection using currently available vrt file.
    obj = s3.Object(f'{STATIC_REPO}', 'WATER_MASK/v0.3/EPSG4326.vrt')
    try:
        obj.get()['Body'].read()
    except Exception:
        errmsg = 'No access to nisar-WATERMASK s3 bucket. Check your AWS credentials ' \
                 'and re-run the code'
        raise ValueError(errmsg)


def apply_margin_polygon(polygon, margin_in_km=5):
    '''
    Convert margin from km to degrees and
    apply to polygon

    Parameters
    ----------
    polygon: shapely.Geometry.Polygon
        Bounding polygon covering the area on the
        ground over which download the DEM
    margin_in_km: np.float
        Buffer in km to add to polygon

    Returns
    ------
    poly_with_margin: shapely.Geometry.box
        Bounding box with margin applied
    '''
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    lat_worst_case = max([lat_min, lat_max])

    # Convert margin from km to degrees
    lat_margin = margin_km_to_deg(margin_in_km)
    lon_margin = margin_km_to_longitude_deg(margin_in_km, lat=lat_worst_case)

    poly_with_margin = box(lon_min - lon_margin, max([lat_min - lat_margin, -90]),
                           lon_max + lon_margin, min([lat_max + lat_margin, 90]))
    return poly_with_margin


def margin_km_to_deg(margin_in_km):
    '''
    Converts a margin value from km to degrees

    Parameters
    ----------
    margin_in_km: np.float
        Margin in km

    Returns
    -------
    margin_in_deg: np.float
        Margin in degrees
    '''
    km_to_deg_at_equator = 1000. / (EARTH_APPROX_CIRCUMFERENCE / 360.)
    margin_in_deg = margin_in_km * km_to_deg_at_equator

    return margin_in_deg


def margin_km_to_longitude_deg(margin_in_km, lat=0):
    '''
    Converts margin from km to degrees as a function of
    latitude

    Parameters
    ----------
    margin_in_km: np.float
        Margin in km
    lat: np.float
        Latitude to use for the conversion

    Returns
    ------
    delta_lon: np.float
        Longitude margin as a result of the conversion
    '''
    delta_lon = (
        180 * 1000 * margin_in_km / (np.pi * EARTH_RADIUS * np.cos(np.pi * lat / 180))
    )
    return delta_lon


def main(opts):
    """Main script to execute water mask staging

    Parameters:
    ----------
    opts : argparse.ArgumentParser
        Argument parser
    """

    # Check if RSLC or bbox are provided
    if (opts.product is None) & (opts.bbox is None):
        errmsg = "Need to provide reference RSLC HDF5 or bounding box. " \
                 "Cannot download water mask"
        raise ValueError(errmsg)

    # Make sure that output file has VRT extension
    if not opts.outfile.lower().endswith('.vrt'):
        err_msg = "water mask output filename extension is not .vrt"
        raise ValueError(err_msg)

    # Determine polygon based on RSLC info or bbox
    poly = determine_polygon(opts.product, opts.bbox)

    # Apply margin to the identified polygon in lat/lon
    poly = apply_margin_polygon(poly, opts.margin)

    # Check dateline crossing. Returns list of polygons
    polys = check_dateline(poly)

    if os.path.isfile(opts.filepath):
        print('Check overlap with user-provided water mask')
        overlap = check_watermask_overlap(opts.filepath, polys)
        if overlap < 75.:
            print('Insufficient water mask coverage. Errors might occur')
        print(f'water mask coverage is {overlap} %')
    else:
        # Check connection to AWS s3 nisar-WATERMASK  ucket
        try:
            check_aws_connection()
        except ImportError:
            import warnings
            warnings.warn('boto3 is required to verify AWS connection '
                          'proceeding without verifying connection')
        # Determine EPSG code
        epsg = determine_projection(polys)
        # Download water mask
        download_watermask(polys, epsg, opts.outfile, opts.version)
        print('Done, water mask store locally')


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)
