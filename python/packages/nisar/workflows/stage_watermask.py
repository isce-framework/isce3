#!/usr/bin/env python3

# WATERMASK staging

import argparse
import os
import re
import warnings

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
    parser.add_argument('-b', '--bbox', type=float, nargs=4,
                        help=('Spatial bounding box as minX, minY, maxX, maxY '
                              'in Spatial Reference System specified by epsg. '
                              'For default epsg=4326, these are west, south, '
                              'east, north bounds in decimal degrees.'))
    parser.add_argument('-e', '--bbox-epsg', type=int, choices=[4326, 3031, 3413],
                        default=4326,
                        help='EPSG code corresponding to the bbox coordinates. '
                             'Must be one of 4326, 3413, or 3031.')
    parser.add_argument('-d', '--watermask-epsg', type=int, choices=[4326, 3031, 3413],
                        default=None,
                        help='EPSG code corresponding to the output Water Mask projection system.'
                             ' Must be one of 4326, 3413, or 3031.'
                             ' If not provided it is assumed to be the same as bbox epsg or'
                             ' the same as the epsg of the input product polygon. The output ' \
                             ' epsg falls back to 4326 if the user requests 3031 or 3413, but the'
                             ' input Water Mask with that epsg code did not cover the specified bbox or'
                             ' the bbox falls outside the valid region of the Water Mask. The Water Mask in 3413 is ' \
                             ' considered valid for latitudes above 70 degrees North and the Water Mask in' \
                             ' epsg 3031 is considered valid below latitude 70 degrees South.')
    parser.add_argument('-v', '--version', type=str, action='store',
                        dest='version', default='0.5',
                        help='Version for water mask')
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
    if ((xmax - xmin > 180.0) or (xmin <= 180.0 <= xmax) or (xmin <= -180.0 <= xmax)):
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

        # The water mask used for NISAR processing has a longitude
        # range [-180, +180]. The current version of gdal.Translate
        # does not allow to perform dateline wrapping. Therefore, coordinates
        # above 180 need to be wrapped down to -180 to match the water mask
        # longitude range
        for polygon_count in range(2):
            x, y = polys[polygon_count].exterior.coords.xy
            if not any([k > 180 for k in x]):  # pylint: disable=use-a-generator
                continue

            # Otherwise, wrap longitude values down to 360 deg
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

        assert (len(polys) == 2)
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def adjust_lat_lon_coordinates(x_min, y_min, x_max, y_max, vrt_filename):
    """Adjust a bbox to fall within the bounds of input Raster (vrt_filename)

    Parameters
    ----------
    x_min: float
        Minimum X coordinate of the bbox
    y_min: float
        Minimum Y coordinate of the bbox
    x_max: float
        Maximum X coordinate of the bbox
    y_max: float
        Maximum Y coordinate of the bbox
    vrt_filename: str
        The VRT filename of the input Raster (Water Mask)

    Returns
    -------
    x_min: float
        Adjusted minimum X coordinate of the bbox
    y_min: float
        Adjusted minimum Y coordinate of the bbox
    x_max: float
        Adjusted maximum X coordinate of the bbox
    y_max: float
        Adjusted maximum Y coordinate of the bbox
    """
    # the original Water Mask and consequently the Water Mask for NISAR
    # has a coverage of -180.0001388888889 to 179.9998611111111 in longitude
    # and a coverage of -89.99986111111112 to 90.00013888888888 in latitude.
    # This non-integer coverage makes it difficult to check if the Water Mask covers
    # the bounding box when the bbox includes the nominal boundaries of the Water Mask.
    # For example when we split a polygon crossing antimeridian to two polygons then
    # we may have a polygon that goes from 179.0 to 180.0 degrees in longitude.
    # In such situation, we may get a coverage error as the Water Mask goes up to 179.9998611111111
    # and does not cover the 180.0. Therefore a workaround is to adjust the bbox
    # to not go over the Water Mask's actual limits for epsg 4326

    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    input_x_min, xres, row_rotation, input_y_max, col_rotation, yres = ds.GetGeoTransform()
    assert row_rotation == 0.0
    assert col_rotation == 0.0
    length = ds.GetRasterBand(1).YSize
    width = ds.GetRasterBand(1).XSize

    input_y_min = input_y_max + length * yres
    input_x_max = input_x_min + width * xres
    input_x_min, input_x_max = sorted((input_x_min, input_x_max))
    input_y_min, input_y_max = sorted((input_y_min, input_y_max))

    x_min = max(x_min, input_x_min)
    x_max = min(x_max, input_x_max)
    y_min = max(y_min, input_y_min)
    y_max = min(y_max, input_y_max)

    return x_min, y_min, x_max, y_max


def determine_polygon(ref_slc, bbox, bbox_epsg):
    """Determine bounding polygon using RSLC radar grid/orbit
    or user-defined bounding box

    Parameters
    ----------
    ref_slc: str
        Filepath to reference RSLC product
    bbox: list, float
        Bounding box with Xmin, Ymin, Xmax, Ymax
        For epsg 4326, it should be in with lat/lon coordinates (decimal degrees)
        in the form of [West, South, East, North]
    bbox_epsg: int
        EPSG code corresponding to the bbox

    Returns
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RSLC perimeter
        or bbox shape on the ground
    epsg: int
        epsg corresponding to the returned polygon.
        The returned epsg is the same input epsg when
        a bounding box passed to the function. When a
        NISAR RSLC product is passed the epsg is assumed
        to be 4326.
    """
    if bbox is not None:
        print('Determine polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        epsg = bbox_epsg

    else:
        print('Determine polygon from RSLC radar grid and orbit')
        poly = get_geo_polygon(ref_slc)
        epsg = 4326

    return poly, epsg


def point2epsg(lon, lat):
    """Return EPSG code based on point lat/lon

    Parameters
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point

    Returns
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

    Parameters
    ----------
    ref_slc: str
        Path to RSLC product to stage the water mask for
    min_height: float
        Global minimum height (in m) for water mask interpolator
    max_height: float
        Global maximum height (in m) for water mask interpolator
    pts_per_edge: float
        Number of points per edge for min/max bounding box computation

    Returns
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RSLC perimeter on the ground
    """
    from isce3.core import LUT2d  # pylint: disable=import-error
    from isce3.geometry import DEMInterpolator, get_geo_perimeter_wkt  # pylint: disable=import-error
    from nisar.products.readers import RSLC  # pylint: disable=import-error

    # Prepare SLC dataset input
    productSlc = RSLC(hdf5file=ref_slc)

    # Extract orbits, radar grid, and doppler for frequency A or B
    orbit = productSlc.getOrbit()
    freq = productSlc.identification.listOfFrequencies[0]
    radar_grid = productSlc.getRadarGrid(frequency=freq)
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

    Parameters
    ----------
    polys: shapely.Geometry.Polygon
        List of shapely Polygons

    Returns
    -------
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
def translate_watermask(vrt_filename, outpath, x_min, x_max, y_min, y_max, epsg):
    """Translate water mask from nisar-WATERMASK bucket. This
       function is decorated to perform retries
       using exponential backoff to make the remote
       call resilient to transient issues stemming
       from network access, authorization and AWS
       throttling (see "Query throttling" section at
       https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html).

    Parameters
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
    epsg: int
        EPSG code representing the projection system of x/y coordinates
    """

    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    srs = osr.SpatialReference(wkt=ds.GetProjection())
    watermask_epsg = int(srs.GetAttrValue("AUTHORITY", 1))
    if epsg != watermask_epsg:
        raise ValueError(
            "Crop bounds must be given "
            f"in the same projection as Water Mask (EPSG:{watermask_epsg}) but they "
            f"were given in EPSG:{epsg} instead.")

    input_x_min, xres, _, input_y_max, _, yres = ds.GetGeoTransform()

    # Declare lambda function to snap min/max X and Y
    # coordinates over the water grid
    snap_coord = lambda val, snap, offset, round_func: round_func(  # noqa: E731
        float(val - offset) / snap) * snap + offset

    # Snap edge coordinates using the raster pixel spacing
    # and starting coordinates. Max values are rounded
    # using np.ceil and min values are rounded with np.floor
    x_min = snap_coord(x_min, xres, input_x_min, np.floor)
    x_max = snap_coord(x_max, xres, input_x_min, np.ceil)
    y_min = snap_coord(y_min, yres, input_y_max, np.floor)
    y_max = snap_coord(y_max, yres, input_y_max, np.ceil)

    gdal.Translate(outpath, ds, format='GTiff',
                   projWin=[x_min, y_max, x_max, y_min])

    # stage_watermask.py takes a bbox as an input. The longitude coordinates
    # of this bbox are unwrapped i.e., range in [0, 360] deg. If the
    # bbox crosses the anti-meridian, the script divides it in two
    # bboxes neighboring the anti-meridian. Here, x_min and x_max
    # represent the min and max longitude coordinates of one of these
    # bboxes. We Add 360 deg if the min longitude of the downloaded water mask
    # tile is < 180 deg i.e., there is a dateline crossing.
    # This ensure that the mosaicked water mask VRT will span a min
    # range of longitudes rather than the full [-180, 180] deg
    if x_min <= -180.0 and watermask_epsg == 4326:
        ds = gdal.Open(outpath, gdal.GA_Update)
        geotransform = list(ds.GetGeoTransform())
        geotransform[0] += 360.0
        ds.SetGeoTransform(tuple(geotransform))

    ds = None


def download_watermask(polys, epsg, outfile, version):
    """Download water mask from nisar-WATERMASK bucket

    Parameters
    ----------
    polys: shapely.geometry.Polygon
        List of shapely polygons
    epsg: int
        The EPSG code corresponding to polys
    outfile: str
        Path to the output WATERMASK file to be staged
    version: str
        Water mask version
    """
    try:
        # Download WATERMASK for each polygon/epsg
        file_prefix = os.path.splitext(outfile)[0]
        watermask_list = []
        for n, poly in enumerate(polys):
            if version == '0.2':
                vrt_filename = f'{WATER_MASK_VSIS3_PATH}/v{version}/watermask.vrt'
            else:
                vrt_filename = f'{WATER_MASK_VSIS3_PATH}/v{version}/EPSG{epsg}.vrt'
            outpath = f'{file_prefix}_{n}.tiff'
            watermask_list.append(outpath)
            xmin, ymin, xmax, ymax = poly.bounds
            if epsg == 4326:
                xmin, ymin, xmax, ymax = adjust_lat_lon_coordinates(xmin, ymin, xmax, ymax, vrt_filename)
            translate_watermask(vrt_filename, outpath, xmin, xmax, ymin, ymax, epsg)

        # Build vrt with downloaded watermasks
        vrt_dataset = gdal.BuildVRT(outfile, watermask_list)

        # Get the WATER description from the LICENSE.txt file using GDAL
        in_license_path = vrt_filename.replace(f'EPSG{epsg}.vrt',
                                               f'EPSG{epsg}/LICENSE.txt')
        license_exists = gdal.VSIStatL(in_license_path) is not None

        if license_exists:
            try:
                water_descr = parse_shortdesc_and_notes(in_license_path)
                vrt_dataset.SetMetadataItem("water_mask_description",
                                            water_descr)

            except Exception as e:
                warnings.warn(
                    f"Found {in_license_path} but could not parse it ({e}). "
                    "Proceeding without metadata."
                )
        else:
            warnings.warn(
                f"No LICENSE.txt found at {in_license_path} "
                f"(expected for WATERMASK v0.4+). Proceeding without metadata."
            )
    except Exception as e:
        errmsg = f'Failed to download NISAR WATERMASK {version} from s3 bucket. ' \
                 f'Maybe {version} is not currently supported.'
        raise ValueError(errmsg) from e


def parse_shortdesc_and_notes(license_path: str,
                        encoding: str = "utf-8") -> str | None:
    """
    Extract the multi-line text of the “Notes” bullet from
    a LICENSE.txt in an S3 bucket via GDAL and return the
    value without the leading "- Notes:" header.

    Parameters
    ----------
    license_path : str
        `/vsis3/.../LICENSE.txt` or any other GDAL compatible URI.
    encoding : str, default "utf-8"
        Text encoding to decode the bytes.

    Returns
    -------
    str | None
        The extracted Notes text (possibly multi-line) with surrounding
        whitespace stripped, or None if no “Notes” bullet is found.
    """
    stat = gdal.VSIStatL(license_path)
    if stat is None:
        raise ValueError(f"No LICENSE.txt found at {license_path}")

    fp = gdal.VSIFOpenL(license_path, "rb")
    if fp is None:
        raise IOError(f"Could not open {license_path} via GDAL")
    try:
        data = gdal.VSIFReadL(1, stat.size, fp)
    finally:
        gdal.VSIFCloseL(fp)

    if not data:
        raise IOError(f"Failed to read any data from {license_path}")

    text = data.decode(encoding, errors="replace")

    if "notes:" not in text.lower():
        warnings.warn(f'No "Notes" field found in {license_path}')
        return None

    NEXT_BULLET = r"(?=^\s*-\s+[^:\n]+:\s*|\Z)"

    def _grab(header: str) -> str | None:
        pat = re.compile(
            rf"^\s*-\s*{re.escape(header)}\s*:\s*(.*?){NEXT_BULLET}",
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        m = pat.search(text)
        return m.group(1).strip() if (m is not None) else None

    shortdesc = _grab("Short description")
    notes = _grab("Notes")

    if not (shortdesc and notes):
        raise ValueError(
            f'Missing "Short description" or "Notes" in {license_path}')

    return ' '.join([shortdesc, notes])


def watermask_covers_bbox_polar_stereo(vrt_filename, x_min, x_max, y_min, y_max, epsg):
    """Check if the Water Mask in polar stereo covers the bbox and if the bbox falls in the valid region of the Water Mask
    For epsg 3413, the Water Mask is considered valid above 65 degrees North
    For epsg 3031, the Water Mask is considered valid below 65 degrees South
    Parameters
    ----------
    vrt_filename: str
        Path to the input VRT file
    x_min: float
        Minimum X coordinate bound of the subwindow, in the coordinate system determined by `epsg`
    x_max: float
        Maximum X coordinate bound of the subwindow, in the coordinate system determined by `epsg`
    y_min: float
        Minimum Y coordinate bound of the subwindow, in the coordinate system determined by `epsg`
    y_max: float
        Maximum Y coordinate bound of the subwindow, in the coordinate system determined by `epsg`
    epsg: int
        The EPSG code corresponding to the input bounding box
    Returns:
    --------
    coverage_status: bool
        A boolean flag indicating if the Water Mask covers the bbox (True) or not (False)
    """
    assert epsg in [3031, 3413]
    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    watermask_epsg = int(srs.GetAttrValue("AUTHORITY", 1))
    if epsg != watermask_epsg:
        raise ValueError(
            "The coordinates of the bounds must be given "
            f"in the same projection as Water Mask (EPSG:{watermask_epsg}) "
            f"but they were given in EPSG:{epsg} instead.")

    input_x_min, xres, row_rotation, input_y_max, col_rotation, yres = ds.GetGeoTransform()
    assert row_rotation == 0.0
    assert col_rotation == 0.0
    length = ds.GetRasterBand(1).YSize
    width = ds.GetRasterBand(1).XSize

    input_y_min = input_y_max + length * yres
    input_x_max = input_x_min + width * xres
    input_x_min, input_x_max = sorted((input_x_min, input_x_max))
    input_y_min, input_y_max = sorted((input_y_min, input_y_max))

    print("Requested bounding box (x_min, x_max, y_min, y_max):", x_min, x_max, y_min, y_max)
    print("Source Water Mask coverage:", input_x_min, input_x_max, input_y_min, input_y_max)
    # Let's check if the Water Mask in the given EPSG that we are looking at
    # covers the coordinates of the snapped bbox
    if (x_min < input_x_min or x_max > input_x_max
            or y_min < input_y_min or y_max > input_y_max):
        # the Water Mask does not cover the bbox
        print("The Water Mask does NOT cover the bbox.")
        return False

    # the Water Mask covers the bbox
    print("The Water Mask covers the bbox.")

    # If epsg in [3031, 3413] we need an extra check
    # to make sure the bbox is in valid region of the Water Mask
    # The Water Mask for NISAR projected to 3031
    # and 3413 are latitude based and therefore the
    # Water Masks in these EPSGs are valid within a circle
    # above latitude 60 N for 3413 (Polar Stereo)  and
    # below 60 S for epsg 3031 (Antarctica). The diagram
    # below shows the situation where the Water Mask is valid
    # within the circle and invalid at the corners of the Water Mask
    # coverage outside the circle. Therefore it is important
    # to check not only the geometrical coverage of the Water Mask
    # and bbox as done above, but also check if the bbox is
    # in the valid region of the Water Mask.
    #
    #            -----------+-------------
    #            |        .-''+''-.  Invalid
    #            |     .-'           '-.  |
    #            |   .'                 '.|
    #            | /                     \|
    #            |                        |
    #            |    VALID    REGION     |
    #            |                        |
    #            | \                     /|
    #            |  '.                 .' |
    #            |   '-.             .-'  |
    #            |       '-..-+-..-'      |
    #            -----------+--------------
    #
    poly = box(x_min, y_min, x_max, y_max)
    poly_lat_lon = transform_bbox_to_latlon(poly, epsg)
    lons, lats = poly_lat_lon.boundary.coords.xy

    if epsg == 3413 and min(lats) > 65:
        watermask_is_valid = True
    elif epsg == 3031 and max(lats) < -65:
        watermask_is_valid = True
    else:
        watermask_is_valid = False

    if not watermask_is_valid:
        print(f"The bbox is not in the valid region of the Water Mask in epsg {epsg}.")
    return watermask_is_valid


def transform_polygon_coords(polys, epsgs):
    """Transform coordinates of polys (list of polygons)
       from epsg 4326 to to target epsgs (list of EPSG codes)

    Parameters
    ----------
    polys: shapely.Geometry.Polygon
        List of shapely polygons or a single shapely polygon
    epsg: list, str
        List of EPSG codes corresponding to
        elements in polys or a single EPSG code
    """

    # If polys or epsgs is a single element, convert them to a list
    if not isinstance(polys, list):
        polys = [polys]
    if not isinstance(epsgs, list):
        epsgs = [epsgs]

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
    poly = Polygon([(min(xmin), min(ymin)), (min(xmin), max(ymax)),
                    (max(xmax), max(ymax)), (max(xmax), min(ymin))])

    return poly


def transform_bbox_to_latlon(poly, epsg):
    """Transform the coordinates of a bounding box
    in a given epsg to the coordinates in lat/lon (epsg 4326)

    Parameters
    ----------
    poly: shapely.Geometry.Polygon
        Input bbox in form of shapely polygon
    epsg: int
        Epsg code corresponding to input poly

    Returns
    -------
    poly: shapely.Geometry.Polygon
        Output bbox in epsg 4326
    """

    epsg_latlon = 4326
    xmin, ymin, xmax, ymax = poly.bounds
    # Create source spatial reference
    source = osr.SpatialReference()
    source.ImportFromEPSG(epsg)

    # Create target spatial reference for epsg 4326
    target = osr.SpatialReference()
    target.ImportFromEPSG(epsg_latlon)

    # Create the coordinate transformation object
    transformer = osr.CoordinateTransformation(source, target)

    ul_latitude, ul_longitude, _ = transformer.TransformPoint(xmin, ymax)
    lr_latitude, lr_longitude, _ = transformer.TransformPoint(xmax, ymin)
    ur_latitude, ur_longitude, _ = transformer.TransformPoint(xmax, ymax)
    ll_latitude, ll_longitude, _ = transformer.TransformPoint(xmin, ymin)

    poly = Polygon([(ul_longitude, ul_latitude), (ur_longitude, ur_latitude),
                    (lr_longitude, lr_latitude), (ll_longitude, ll_latitude)])

    return poly


def check_watermask_overlap(watermaskFilepath, polys):
    """Evaluate overlap between user-provided WATERMASK
       and WATERMASK that stage_watermask.py would download
       based on RSLC or bbox provided information

    Parameters
    ----------
    watermaskFilepath: str
        Filepath to the user-provided WATERMASK
    polys: shapely.geometry.Polygon
        List of polygons computed from RSLC or bbox

    Returns
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
        polys = [transform_polygon_coords(polys, epsg)]

    perc_area = 0
    for poly in polys:
        perc_area += (poly.intersection(poly_watermask).area / poly.area) * 100
    return perc_area


def check_aws_connection(version):
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
    obj = s3.Object(f'{STATIC_REPO}', f'WATER_MASK/v{version}/EPSG4326.vrt')
    try:
        obj.get()['Body'].read()
    except Exception:
        errmsg = 'No access to nisar-WATERMASK s3 bucket. Check your AWS credentials ' \
                 'and re-run the code'
        raise ValueError(errmsg)


def apply_margin_to_geographic_box(polygon, margin_in_km=5):
    '''
    Assuming the polygon is in epsg 4326
    Convert margin from km to degrees and
    apply to polygon

    Parameters
    ----------
    polygon: shapely.Geometry.Polygon
        Bounding polygon covering the area on the
        ground over which download the Water Mask
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

    if lon_max - lon_min > 180:
        lon_min, lon_max = lon_max, lon_min

    poly_with_margin = box(lon_min - lon_margin, max([lat_min - lat_margin, -90]),
                           lon_max + lon_margin, min([lat_max + lat_margin, 90]))
    return poly_with_margin


def apply_margin_to_projected_box(polygon, margin_in_km=5):
    '''
    Assuming the polygon is in UTM or Polar Stereo
    add a margin in km to the polygon
    Parameters
    ----------
    polygon: shapely.Geometry.Polygon
        Bounding polygon covering the area on the
        ground to download the Water Mask
    margin_in_km: np.float
        Buffer in km to add to the polygon

    Returns
    ------
    poly_with_margin: shapely.Geometry.box
        Bounding box with margin applied
    '''
    x_min, y_min, x_max, y_max = polygon.bounds
    assert x_max >= x_min
    assert y_max >= y_min
    margin_in_meters = margin_in_km*1000.0
    poly_with_margin = box(x_min - margin_in_meters, y_min-margin_in_meters,
                           x_max + margin_in_meters, y_max + margin_in_meters)
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

    Parameters
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

    # Determine polygon based on RSLC info or bbox. Priority is given to bbox
    # and the returned epsg is the input epsg. If bbox is not provided but an
    # RSLC product provided, then epsg is assumed to be 4326
    poly, bbox_epsg = determine_polygon(opts.product, opts.bbox, opts.bbox_epsg)

    # the epsg of the output Water Mask if not specified is assumed to be
    # the same as the epsg of the bounding box
    watermask_epsg = opts.watermask_epsg or bbox_epsg

    # The Water Mask only contains three EPSG codes
    valid_epsg_values = {4326, 3413, 3031}
    if bbox_epsg not in valid_epsg_values or watermask_epsg not in valid_epsg_values:
        raise ValueError("Both bbox_epsg and watermask_epsg must be one of 4326, 3413, or 3031.")

    if bbox_epsg != watermask_epsg:
        if bbox_epsg != 4326:
            # Users are allowed to provide bbox_epsg in lat/lon (epsg 4326) and ask for a Water Mask
            # in watermask_epsg different than bbox_epsg (i.e., 3413, 3031). However, we do
            # not encourage the opposite.
            raise ValueError("If bbox epsg and watermask_epsg are different, then bbox epsg cannot be 3413 or 3031.")
        # Transform the polygon in 4326 to polar stereo
        poly = transform_polygon_coords(poly, watermask_epsg)

    # At this point the polygon is at watermask_epsg projection.
    # So no more use for bbox_epsg from this point

    if watermask_epsg == 4326:
        # Apply margin to the identified polygon in lat/lon
        poly = apply_margin_to_geographic_box(poly, opts.margin)

        # Check dateline crossing. Returns list of polygons
        polys = check_dateline(poly)
    else:
        # epsg is 3031 or 3413
        # Apply margin to the identified polygon in polar stereo
        poly = apply_margin_to_projected_box(poly, opts.margin)
        polys = [poly]

    # check if the Water Mask in epsg 3031 or 3413 covers the bbox and if the bbox falls
    # to the valid part of the Water Mask. If the Water Mask does not cover the bbox, or if
    # the bbox is not in the valid part of the Water Mask, transform the bbox to
    # epsg 4326 and update watermask_epsg to be 4326
    if watermask_epsg in [3031, 3413]:
        vrt_filename = \
            f'{WATER_MASK_VSIS3_PATH}/v{opts.version}/EPSG{watermask_epsg}.vrt'
        xmin, ymin, xmax, ymax = polys[0].bounds
        covers_bbox = watermask_covers_bbox_polar_stereo(vrt_filename, xmin, xmax, ymin, ymax, watermask_epsg)
        if not covers_bbox:
            print(f"Water Mask in epsg {watermask_epsg} does not cover bbox. Transforming to lat/lon")
            print("Transforming the bbox to lat/lon (epsg 4326) and download a Water Mask in epsg 4326 ")
            poly = transform_bbox_to_latlon(polys[0], watermask_epsg)
            watermask_epsg = 4326
            polys = check_dateline(poly)

    if os.path.isfile(opts.filepath):
        print('Check overlap with user-provided water mask')
        overlap = check_watermask_overlap(opts.filepath, polys)
        if overlap < 75.:
            print('Insufficient water mask coverage. Errors might occur')
        print(f'water mask coverage is {overlap} %')
    else:
        # Check connection to AWS s3 nisar-WATERMASK bucket
        try:
            check_aws_connection(opts.version)
        except ImportError:
            import warnings
            warnings.warn('boto3 is required to verify AWS connection '
                          'proceeding without verifying connection')

        # Download water mask
        download_watermask(polys, watermask_epsg, opts.outfile, opts.version)
        print('Done, water mask store locally')


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)
