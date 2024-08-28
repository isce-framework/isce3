from isce3.geometry import DEMInterpolator
import logging
import numpy as np
from osgeo import ogr, osr, gdal
import textwrap
from typing import Union
import shapely
from shapely import Polygon, MultiPolygon
import shapely.affinity
import shapely.wkt
try:
    import shapely.plotting
    import matplotlib.pyplot as plt
except ImportError:
    can_plot = False
else:
    can_plot = True


log = logging.getLogger("isce3.geometry.polygons")
osr.UseExceptions()

def get_dem_boundary_polygon(dem: DEMInterpolator, n=11):
    """
    Compute a polygon representing perimeter of a DEM.

    Parameters 
    ---------
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model.  Must have a loaded raster.
    n : int, optional
        Number of points along each edge.

    Returns
    -------
    polygon : osgeo.ogr.Geometry
        GDAL polygon object, in same spatial reference system as DEM.
    """
    if n < 2:
        raise ValueError(f"Require at least two points per edge, requested={n}")
    if not dem.have_raster:
        raise ValueError("Requested boundary polygon for DEM but no "
            "raster was loaded.")
    # Figure out corners of DEM and put them in counter-clockwise order
    # in order to conform with OGR standard.  Be careful to handle negative
    # step sizes.
    x0 = dem.x_start
    x1 = x0 + dem.delta_x * dem.width
    x0, x1 = sorted((x0, x1))
    
    y0 = dem.y_start
    y1 = y0 + dem.delta_y * dem.length
    y0, y1 = sorted((y0, y1))

    corners = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0]])

    # Generate the four edges, with n points per side.
    t = np.arange(n - 1) / (n - 1)
    boundary = []
    start = corners[0]
    for end in corners[1:]:
        edge = start + (end - start) * t[:,None]
        boundary.extend(edge)
        start = end
    boundary.append([x0, y0])

    # Construct the OGR polygon object.
    polygon = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in boundary:
        # Use mean_height rather than try to interpolate along the edge because
        # interpolation kernel needs finite support and will just return
        # ref_height for points right on the edge.
        ring.AddPoint(*point, dem.mean_height)
    polygon.AddGeometry(ring)

    #XXX DEMInterpolator doesn't keep track of the full SRS of the DEM file,
    #XXX which can include a data axis to CRS axis mapping flip.  Assume that
    #XXX axes are flipped for EPSG:4326 (longitude is the x-axis).
    if dem.epsg_code == 4326:
        srs = get_srs_lonlat()
    else:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(dem.epsg_code)
    polygon.AssignSpatialReference(srs)
    return polygon


def plot_shapely_polygon_km(poly, add_points=False, **opts):
    """
    Scale all coordinates of a polygon by 1/1000 and call shapely plot_polygon
    on result.

    Parameters
    ----------
    poly : shapely.Polygon
        Polygon object, with coordinates in meters.
    add_points : bool, optional
        Whether to add markers for all vertices.
    opts : kwargs
        Options forwarded to plot_polygon
    
    Returns
    -------
    Result of shapely.plotting.plot_polygon or None if polygon is empty.
    """
    if poly.is_empty:
        return
    scaled = shapely.affinity.scale(poly, xfact=0.001, yfact=0.001, zfact=0.001,
        origin=(0.0, 0.0, 0.0))
    return shapely.plotting.plot_polygon(scaled, add_points=add_points, **opts)


def plot_dem_overlap(swath_polygon, dem_polygon):
    """
    Plot radar swath and DEM on a map.

    Parameters
    ----------
    swath_polygon : osgeo.ogr.Geometry
        OGR polygon representing the radar swath.
    dem_polygon : osgeo.ogr.Geometry
        OGR polygon representing the DEM.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    swath = shapely.wkt.loads(swath_polygon.ExportToWkt())
    dem = shapely.wkt.loads(dem_polygon.ExportToWkt())
    swath_inside_dem = swath & dem
    swath_outside_dem = swath - dem
    fig = plt.figure(dpi=180)
    plot_shapely_polygon_km(dem, color="tab:blue", label="DEM")
    plot_shapely_polygon_km(swath_inside_dem, color="tab:green", label="Inside")
    plot_shapely_polygon_km(swath_outside_dem, color="tab:red", label="Outside")
    plt.legend(loc="best")
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    title = swath_polygon.GetSpatialReference().ExportToProj4()
    plt.title("\n".join(textwrap.wrap(title, width=40)))
    plt.tight_layout()
    return fig


def get_srs_lonlat() -> osr.SpatialReference:
    """
    Get GDAL spatial reference system corresponding to points in
    (longitude, latitude) order in degrees.
    """
    # Careful because GDAL flipped the coordinate order in version 3.
    srs_lonlat = osr.SpatialReference()
    srs_lonlat.ImportFromEPSG(4326)
    srs_lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs_lonlat


def get_srs_local(longitude: float, latitude: float) -> osr.SpatialReference:
    """
    Get a Lambert azimuthal equal area spatial reference system centered on the
    specified point.
    """
    srs_local = osr.SpatialReference()
    srs_local.SetLAEA(latitude, longitude, 0, 0)
    srs_local.SetLinearUnits("Meter", 1.0)
    return srs_local


def compute_dem_overlap(swath_polygon_wkt: str, dem: DEMInterpolator, plot=None):
    """
    Determine fraction of swath that falls out the DEM coverage area.

    Parameters
    ----------
    swath_polygon_wkt : str
        Well-known text (WKT) representation of the radar swath polygon, such
        as the contents of the boundingPolygon dataset in NISAR products.
        (X, Y) coordinates should correspond to (longitude, latitude) in degrees
        on the WGS84 ellipsoid (EPSG:4326).
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model (DEM) to compare against.
    plot : file_like, optional
        File like object to save a plot to.  If the required plotting libraries
        are not available, then a warning will be logged and no plot will be
        generated.

    Returns
    -------
    area_outside : float
        Area of the swath that falls outside the DEM coverage area, expressed
        as a fraction of the total swath area.
    """
    if not dem.have_raster:
        return 0.0

    # Create radar swath polygon
    swath_polygon = ogr.CreateGeometryFromWkt(swath_polygon_wkt)
    srs_lonlat = get_srs_lonlat()
    swath_polygon.AssignSpatialReference(srs_lonlat)

    # Create DEM polygon
    dem_polygon = get_dem_boundary_polygon(dem)
    srs_dem = dem_polygon.GetSpatialReference()

    # Set up a local map projection to avoid issues with pole or dateline
    # crossing.  Choose Lambert equal area since we care about areas, and we
    # assume density of points is high enough and overall shape is small enough
    # that distortion of shape is negligible.
    lon, lat = swath_polygon.GetGeometryRef(0).GetPoint(0)[:2]
    srs_local = get_srs_local(lon, lat)

    # Transform polygons to this common coordinate system.
    swath_polygon.Transform(osr.CoordinateTransformation(srs_lonlat, srs_local))
    dem_polygon.Transform(osr.CoordinateTransformation(srs_dem, srs_local))

    if plot is not None:
        if can_plot:
            plot_dem_overlap(swath_polygon, dem_polygon).savefig(plot)
        else:
            log.warning("Requested DEM overlap plot but the required plotting "
                "modules could not be imported.")

    swath_outside_dem = swath_polygon.Difference(dem_polygon)
    return swath_outside_dem.GetArea() / swath_polygon.GetArea()


# Added to support track/frame & bounding polygon intersection.

def shapely2ogr_polygon(poly: Union[Polygon, MultiPolygon],
                        h: float = 0.0) -> ogr.Geometry:
    """
    Convert a shapely polygon object to an OGR polygon object.

    Parameters
    ----------
    poly : Union[shapely.Polygon, shapely.MultiPolygon]
        Input shapely Polygon
    h : float
        Height to use if polygon doesn't have Z values.

    Returns
    -------
    polygon : ogr.Geometry
        Output OGR polygon (or multi-polygon)
    """
    if isinstance(poly, MultiPolygon):
        out = ogr.Geometry(ogr.wkbMultiPolygon)
        for p in poly.geoms:
            out.AddGeometry(shapely2ogr_polygon(p))
        return out

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in shapely.get_coordinates(poly, include_z=True):
        lon, lat, z = point
        if not poly.has_z:
            z = h
        ring.AddPoint(lon, lat, z)

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def compute_polygon_overlap(poly1: Union[Polygon, MultiPolygon],
                            poly2: Union[Polygon, MultiPolygon]) -> float:
    """
    Take two lon/lat polygons, convert to a local projection, and calculate the
    amount of overlap.

    Parameters
    ----------
    poly1 : Union[shapely.Polygon, shapely.MultiPolygon]
        Polygon where (x, y) correspond to (longitude, latitude) in degrees.
    poly2 : Union[shapely.Polygon, shapely.MultiPolygon]
        Polygon where (x, y) correspond to (longitude, latitude) in degrees.

    Returns
    -------
    area_fraction : float
        Area of intersection of poly1 & poly2 normalized by the area of poly1
    """
    # convert to GDAL format
    poly1 = shapely2ogr_polygon(poly1)
    poly2 = shapely2ogr_polygon(poly2)

    # assign SRS
    srs_lonlat = get_srs_lonlat()
    poly1.AssignSpatialReference(srs_lonlat)
    poly2.AssignSpatialReference(srs_lonlat)

    # convert to a local LAEA projection to avoid issues with dateline or pole
    if poly1.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        ring = poly1.GetGeometryRef(0).GetGeometryRef(0)
    else:
        ring = poly1.GetGeometryRef(0)
    lon, lat = ring.GetPoint(0)[:2]
    srs_local = get_srs_local(lon, lat)
    xform = osr.CoordinateTransformation(srs_lonlat, srs_local)
    poly1.Transform(xform)
    poly2.Transform(xform)

    return poly1.Intersection(poly2).GetArea() / poly1.GetArea()
