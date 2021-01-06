# DEM staging

import argparse
import os
from osgeo import gdal
import numpy as np
import shapely.wkt
from osgeo import ogr
from osgeo import osr
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.geometry import Point


def cmdLineParse():
    """
     Command line parser
    """
    parser = argparse.ArgumentParser(description="""
                                     Stage and verify DEM for processing. """,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required Arguments
    parser.add_argument('-p', '--product', type=str, action='store',
                        help='Input reference RSLC HDF5 product')
    # Optional Arguments
    parser.add_argument('-o', '--output', type=str, action='store',
                        default='dem.tif', dest='outfile',
                        help='Output DEM filepath (GeoTiff format).')
    parser.add_argument('-f', '--path', type=str, action='store',
                        dest='filepath', default='file',
                        help='Filepath to user DEM.')
    parser.add_argument('-t', '--track', action='store',
                        help='Filepath for track-frame database')
    parser.add_argument('-m', '--margin', type=int, action='store',
                        default=5, help='Margin for DEM bounding box (km)')
    parser.add_argument('-b', '--bbox', type=float, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in latitude/longitude (WSEN, decimal degrees)')

    # Parse and return
    return parser.parse_args()


def determine_polygon(ref_slc, bbox=None):
    """
     Determine RSLC polygon on the ground based on
     RSLC radar grid/orbits or bounding box
    """
    if bbox is not None:
        print('Determine polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        print('Determine polygon from RSLC radar grid and orbit')
        poly = get_geo_polygon(ref_slc)

    return poly


def point2epsg(lon, lat):
    """
     Return EPSG code base on a point
     latitude/longitude coordinates
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
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
    """
     Create polygon (EPSG:4326)) using RSLC radar grid and orbits
    """
    from pybind_isce3.geometry import get_geo_perimeter_wkt
    from pybind_isce3.geometry import DEMInterpolator
    from pybind_isce3.core import LUT2d
    from pybind_nisar.products.readers import SLC

    # Prepare SLC dataset input
    productSlc = SLC(hdf5file=ref_slc)

    # Extract orbits, radar grid, and doppler for frequency A
    orbit = productSlc.getOrbit()
    radar_grid = productSlc.getRadarGrid(frequency='A')
    doppler = LUT2d()

    # Get min and max global height DEM interpolators
    dem_min = DEMInterpolator(height=min_height)
    dem_max = DEMInterpolator(height=max_height)

    # Get min and max bounding boxes
    box_min = get_geo_perimeter_wkt(radar_grid, orbit, doppler,
                                    dem_min, pts_per_edge)
    box_max = get_geo_perimeter_wkt(radar_grid, orbit, doppler,
                                    dem_max, pts_per_edge)

    # Determine minimum and maximum polygons
    poly_min = shapely.wkt.loads(box_min)
    poly_max = shapely.wkt.loads(box_max)

    # Get polygon from intersection of poly_min and poly_max
    poly = poly_min | poly_max

    return poly


def determine_projection(poly, track_frame):
    """
     Determine EPSG code of the area covered by
     poly and compare to that of track frame database
     EPSG is computed for a regular grid of points
     in poly. Final EPSG selected based on majority criteria
    """

    # Make a regular grid based on poly min/max latitude longitude
    minX, minY, maxX, maxY = poly.bounds
    xx, yy = np.meshgrid(np.linspace(minX, maxX, 250), np.linspace(minY, maxY, 250))
    x = xx.flatten()
    y = yy.flatten()

    # Query to determine the zone
    zones = []
    for lx, ly in zip(x, y):
        # Create a point with grid coordinates
        pp = Point(lx, ly)

        # If Point is in poly, compute EPSG
        if pp.within(poly):
            zones.append(point2epsg(lx, ly))
        else:
            pass

    # Count different EPSGs
    vals, counts = np.unique(zones, return_counts=True)

    # Get the ESPG for Poly
    epsg_per = vals[np.argmax(counts)]

    if track_frame is None:
        print('Assigning EPSG from polygon')
        return epsg_per
    else:
        print('Comparing EPSG from polygon with track frame database')

        # Open Track Frame database using OGR
        dataSource = ogr.Open(track_frame, 0)
        layer = dataSource.GetLayer('frames')
        area = []
        epsg_dummy = []

        # Filter the Track frame data base based on Polygon Bounding box
        layer.SetSpatialFilterRect(*poly.bounds)

        # Compute area of intersection between SLC poly and track frames
        for feature in layer:
            frame_poly = shapely.wkt.loads(feature.geometry().ExportToWkt())
            area.append(poly.intersection(frame_poly).area)
            epsg_dummy.append(feature.GetField("epsg"))

        epsg_track = epsg_dummy[np.argmax(area)]

        if epsg_per != epsg_track:
            print('EPSG from polygon differs from EPSG in track frame')
            print('Assigning EPSG from track frame database')
            return epsg_track
        else:
            print('EPSG from polygon matches EPSG from track frame')
            return epsg_per


def download_dem(poly, epsg, margin, outfile):
    """
     Download DEM from AWS S3 bucket
     using bounding box end EPSG coordinates
    """
    # Transform poly coords in target epsg
    minX, maxX, minY, maxY = transform_polygon_coords(poly, epsg)

    # Pad with margins
    minX = minX - margin * 1000
    minY = minY - margin * 1000
    maxX = maxX + margin * 1000
    maxY = maxY + margin * 1000

    # Determine vrt filepath
    vrt_filename = return_dem_filepath(epsg)

    print(f"S3 bucket DEM filepath: ", vrt_filename)

    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    gdal.Warp(outfile, ds, format='GTiff',
              outputBounds=[minX, minY, maxX, maxY], multithread=True)
    ds = None


def return_dem_filepath(epsg, ref_slc=None,
                        track_frame=None):
    """
       Identify and return the path to the AWS S3 NISARDEM
       bucket containing the DEM to download
    """
    if ref_slc is not None:
        # Determine polygon based on RSLC radar grid and orbits
        poly = get_geo_polygon(ref_slc)
        # Determine epsg
        epsg = determine_projection(poly, track_frame)

    # Get the DEM vrt Filename
    vrt_filename = '/vsis3/nisar-dem/EPSG' + str(epsg) + \
                   '/EPSG' + str(epsg) + '.vrt'

    return vrt_filename


def transform_polygon_coords(poly, epsg):
    """
      Transform poly perimeter coordinates to target epsg
      and determine min/max
    """

    # Transform each point of the perimeter in target EPSG coordinates
    llh = osr.SpatialReference()
    llh.ImportFromEPSG(4326)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(int(epsg))
    trans = osr.CoordinateTransformation(llh, tgt)
    x, y = poly.exterior.coords.xy
    
    tgt_x = []
    tgt_y = []

    for lx, ly in zip(x, y):
        dummy_x, dummy_y, dummy_z = trans.TransformPoint(ly, lx, 0)
        tgt_x.append(dummy_x)
        tgt_y.append(dummy_y)

    minX = min(tgt_x)
    maxX = max(tgt_x)
    minY = min(tgt_y)
    maxY = max(tgt_y)

    return [minX, maxX, minY, maxY]


def check_dem_overlap(DEMFilepath, poly):
    """
     Evaluate overlap between provided DEM
     and area covered by the RSLC geo perimeter
    """
    from pybind_isce3.io import Raster

    # Get local DEM edge coordinates
    DEM = Raster(DEMFilepath)
    ulx, xres, xskew, uly, yskew, yres = DEM.get_geotransform()
    lrx = ulx + (DEM.width * xres)
    lry = uly + (DEM.length * yres)
    poly_dem = Polygon([(ulx, uly), (ulx, lry), (lrx, lry), (lrx, uly)])

    if DEM.get_epsg() != 4326:
        minX, maxX, minY, maxY = transform_polygon_coords(poly, DEM.get_epsg())
        poly = Polygon([(minX, minY), (minX, maxY),
                        (maxX, maxY), (maxX, minY)])

    perc_area = (poly.intersection(poly_dem).area / poly.area) * 100
    return perc_area


def check_aws_connection():
    """
       Check if user can read files from s3://nisar-dem
       and throw an exception if not possible
    """
    import boto3

    s3 = boto3.resource('s3')
    obj = s3.Object('nisar-dem', 'EPSG32610/EPSG32610.vrt')

    try:
        body = obj.get()['Body'].read()
    except:
        errmsg = 'No access to nisar-dem s3 bucket. Check your AWS credentials' \
                 'and re-run the code'
        raise ValueError(errmsg)


def main(opts):
    # Check if RSLC or bbox are provided
    if (opts.product is None) & (opts.bbox is None):
        errmsg = "Need to provide reference RSLC HDF5 or bounding box. " \
                 "Cannot download DEM"
        raise ValueError(errmsg)

    # Determine polygon
    poly = determine_polygon(opts.product, opts.bbox)

    if os.path.isfile(opts.filepath):
        print('Check overlap with user-provided DEM')
        overlap = check_dem_overlap(opts.filepath, poly)
        if overlap < 75.:
            print('Insufficient DEM coverage. Errors might occur')
        print(f'DEM coverage is {overlap} %')

    else:
        # Check connection to AWS s3 nisar-dem bucket
        try:
            check_aws_connection()
        except ImportError:
            import warnings
            warnings.warn('boto3 is require to verify AWS connection'
                          'proceeding without verifying connection')
        print('Determine EPSG code')
        epsg = determine_projection(poly, opts.track)
        print('Download DEM')
        download_dem(poly, epsg, opts.margin, opts.outfile)
        print('Done, DEM store locally')


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)
