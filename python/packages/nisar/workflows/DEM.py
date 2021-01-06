# Suite of functionalities for DEM staging

import argparse
import os
import json
from osgeo import gdal
import numpy as np
from osgeo import ogr
from osgeo import osr
from shapely.geometry.polygon import LinearRing
from shapely.geometry.polygon import Polygon


def cmdLineParse():
    """
     Command line parser
    """
    parser = argparse.ArgumentParser(description="""
                                     Stage and verify DEM for processing. """,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required Arguments
    parser.add_argument('-p', '--product', type=str, required=True,
                        help='Input reference HDF5 product')
    # Optional Arguments
    parser.add_argument('-o', '--output', type=str, action='store',
                        default='dem.tif', dest='outfile',
                        help='Output file name.')
    parser.add_argument('-f', '--path', type=str, action='store',
                        dest='filepath', default='file',
                        help='Filepath to available DEM.')
    parser.add_argument('-t', '--track', action='store',
                        help='Filepath for track-frame database')
    parser.add_argument('-m', '--margin', type=int, action='store',
                        default=5000, help='Margin for DEM bounding box (m)')
    parser.add_argument('-b', '--bbox', type=int, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in latitude/longitude (south north west east)')

    # Parse and return
    return parser.parse_args()


def LatLon2UTM(lon, lat):
    """
       Latitude/Longitude to UTM conversion
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177)/6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177)/6.0))
    else:
        raise ValueError(
            'Could not determine projection for {0},{1}'.format(lat, lon))


def determine_perimeter(opts):
    """
     Determine perimeter for DEM staging
    """
    from nisar.products.readers import SLC
    from isce3.geometry import deminterpolator
    from isce3.geometry import getGeoPerimeter
    from isce3.core import projection

    if opts.bbox is not None:
        print('Determine perimeter from bounding box')
        lat = opts.bbox[0:2]
        lon = opts.bbox[2:4]
        ring = LinearRing([(lon[0], lat[0]), (lon[0], lat[1]),
                           (lon[1], lat[1]), (lon[1], lat[0])])
    else:
        print('Determine perimeter from SLC radar grid')

    # Prepare SLC dataset input
    productSlc = SLC(hdf5file=opts.product)

    # Extract orbits and radar Grid parameters
    orbit = productSlc.getOrbit()
    radarGrid = productSlc.getRadarGrid()

    # Minimum and Maximum global heights
    dem_min = deminterpolator(height=-500.)
    dem_max = deminterpolator(height=9000.)

    # Get Minimum and Maximum bounding boxes
    epsg = projection(epsg=4326)
    box_min = getGeoPerimeter(radarGrid, orbit,
                              epsg, dem=dem_min, pointsPerEdge=5)
    box_max = getGeoPerimeter(radarGrid, orbit,
                              epsg, dem=dem_max, pointsPerEdge=5)

    dummy = json.loads(box_min)['coordinates'] + \
        json.loads(box_max)['coordinates']
    ring = LinearRing(dummy)

    return ring


def determine_projection(ring, track_frame):
    """
        Determine projection based on perimeter
        and compare with track frame database
    """
    # Split coordinates
    x, y = ring.coords.xy

    # Query to determine the zone
    zones = []

    for lx, ly in zip(x, y):
        zones.append(LatLon2UTM(lx, ly))

    vals, counts = np.unique(zones, return_counts=True)

    # Projection from perimeter
    epsg_per = vals[np.argmax(counts)]

    # Get Bounding Box from ring
    minX, minY, maxX, maxY = ring.bounds

    if track_frame:
        print('Comparing EPSG from perimeter with track frame database')

        # Open Track Frame database using OGR
        dataSource = ogr.Open(track_frame, 0)  # Do not overwrite
        layer = dataSource.GetLayer('frames')

        # Filter the Track frame data base based on Bounding box
        layer.SetSpatialFilterRect(minX, minY, maxX, maxY)
        hasSeaIce = []
        epsg_dummy = []

        for feature in layer:
            hasSeaIce.append(feature.GetField("hasSeaIce"))
            epsg_dummy.append(feature.GetField("epsg"))
            vals, counts = np.unique(epsg_dummy, return_counts=True)
            epsg_track = vals[np.argmax(counts)]

        epsg = epsg_per
        if epsg_per != epsg_track:
            print('EPSG from perimeter does not match EPSG from trackframe')
            print('Assigning EPSG from trackframe database')
    else:
        print('Track Frame not provided, perimeter from EPSG')
        epsg = epsg_per

    return epsg


def getBbox(ring, epsg):
    """
       Get the Min/Max bounding box
    """
    # Transform each point of the perimeter in target EPSG coordinates
    llh = osr.SpatialReference()
    llh.ImportFromEPSG(4326)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(int(epsg))
    trans = osr.CoordinateTransformation(llh, tgt)
    x, y = ring.coords.xy
    tgt_x = []
    tgt_y = []

    for k in range(0, len(x)):
        dummy_x, dummy_y, dummy_z = trans.TransformPoint(y[k], x[k], 0)
        tgt_x.append(dummy_x)
        tgt_y.append(dummy_y)

    minX = min(tgt_x)
    maxX = max(tgt_x)
    minY = min(tgt_y)
    maxY = max(tgt_y)

    return [minX, maxX, minY, maxY]


def check_dem_overlap(opts, ring):
    """
       Evaluate DEM overlap between existing and downloadable DEM
    """
    from isce3.io import raster

    # Get local DEM edge coordinates
    mm = opts.margin
    DEM = raster(filename=opts.filepath)
    ulx, xres, xskew, uly, yskew, yres = DEM.GeoTransform
    lrx = ulx + (DEM.width * xres)
    lry = uly + (DEM.length * yres)
    minX, maxX, minY, maxY = getBbox(ring, DEM.EPSG)

    # Create the Polygons for both local and downloadable DEM
    Poly_dem = Polygon([(ulx, uly), (ulx, lry), (lrx, lry), (lrx, uly)])
    Poly_ring = Polygon([(minX-mm, minY-mm), (minX-mm, maxY+mm),
                         (maxX+mm, maxY+mm), (maxX+mm, minY-mm)])
    perc_area = (Poly_ring.intersection(Poly_dem).area/Poly_ring.area)*100

    return perc_area


def download_dem(ring, epsg, margin, outfile, track_frame):
    """
       Download a DEM from the S3 bucket based on bounding box and EPSG
    """
    # Get bounding box
    minX, maxX, minY, maxY = getBbox(ring, epsg)

    # Pad with margins
    minX = minX - margin
    minY = minY - margin
    maxX = maxX + margin
    maxY = maxY + margin

    # Cut the identified raster and save locally
    vrt_filename = return_dem_filepath(ring=ring)

    print(f"S3 bucket DEM filepath: ", vrt_filename)

    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)
    outTileName = os.path.join(outfile)
    OutTileName = gdal.Warp(outTileName, ds, format='GTiff', outputBounds=[
                            minX, minY, maxX, maxY], multithread=True)
    ds = None


def return_dem_filepath(orbit=None, radarGrid=None,
                        ring=None, track_frame=None):
    """
       Identify and return the path to the S3 NISARDEM
       bucket containing the DEM to download
    """
    from isce3.geometry import deminterpolator
    from isce3.geometry import getGeoPerimeter
    from isce3.core import projection

    if ring:
        epsg = determine_projection(ring, track_frame)
    else:
        # Minimum and maximum global height
        dem_min = deminterpolator(height=-500)
        dem_max = deminterpolator(height=9000.)

        # Get minimum and maximum bounding boxes
        epsg = projection(epsg=4326)
        box_min = getGeoPerimeter(radarGrid, orbit, epsg,
                                  dem=dem_min, pointsPerEdge=5)
        box_max = getGeoPerimeter(radarGrid, orbit, epsg,
                                  dem=dem_max, pointsPerEdge=5)

        dummy = json.loads(box_min)['coordinates'] + \
                json.loads(box_max)['coordinates']
        ring = LinearRing(dummy)

        # Determine epsg
        epsg = determine_projection(ring, track_frame)

    # Get the DEM vrt Filename
    vrt_filename = '/vsis3/nisar-dem/EPSG' + str(epsg) + \
                   '/EPSG' + str(epsg) + '.vrt'

    return vrt_filename


def main(opts):

    # If DEM is provided, check overlap but not download
    if os.path.isfile(opts.filepath):
        print('DEM already exists')
        ring = determine_perimeter(opts)
        print('Checking DEM overlap')
        overlap = check_dem_overlap(opts, ring)
        print('DEM overlap is ', overlap)
        if overlap < 75.:
            print('Insuffient DEM coverage errors might occur')
        else:
            print('Sufficient DEM coverage')
    # If output file does not exist, download DEM
    else:
        print('Determining DEM perimeter')
        ring = determine_perimeter(opts)
        print('Determining projection code')
        epsg = determine_projection(ring, opts.track)
        print('Downloading DEM')
        download_dem(ring, epsg, opts.margin, opts.outfile, opts.track)
        print('Done, DEM stored locally')


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# End of File
