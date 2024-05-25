#!python3
from .sar_duration import get_sar_duration
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
import isce3
import numpy as np
import shapely
import shapely.affinity  # need explicit import for older versions


@dataclass
class RadarPoint:
    """
    A point in radar coordinates

    Parameters
    ----------
    time : float
        Time in seconds since some reference epoch.
    range : float
        Range in meters relative to some reference orbit.
    """
    time: float
    range: float


@dataclass
class RadarBoundingBox:
    """
    A bounding box extent in radar coordinates.

    Parameters
    ----------
    first : RadarPoint
        Radar coordinate of smallest time and range
    last : RadarPoint
        Radar coordinate of largest time and range
    """
    first: RadarPoint
    last: RadarPoint

    def __post_init__(self):
        if self.first.time > self.last.time:
            raise ValueError("require first.time <= last.time")
        if self.first.range > self.last.range:
            raise ValueError("require first.range <= last.range")

    @property
    def centroid(self) -> RadarPoint:
        """Return the point at the center of the region."""
        return RadarPoint((first.time + last.time) / 2,
            (first.range + last.range) / 2)


def get_raw_sub_swath_polygons(*, raw_bbox_lists, chirp_durations, orbit,
                               wavelength, azres, prf,
                               ellipsoid=isce3.core.Ellipsoid(),
                               convolution_mode="valid",
                               allowed_azimuth_gap=2):
    """
    Get sub-swath polygons in radar coordinates (native Doppler), accounting for
    areas gained or lost due to convolution in range & azimuth and cases of
    mixed-mode processing.

    Parameters
    ----------
    raw_bbox_lists : list[list[RadarBoundingBox]]
        Bounding boxes of all subswaths for each raw data file/observation.
        Azimuth times should be specified in seconds relative to the orbit
        reference epoch.
    chirp_durations : list[float]
        Duration of transmit chirp for each raw data file, in seconds.
    orbit : isce3.core.Orbit
        Radar trajectory.
    wavelength : float
        Radar wavelength in meters.
    azres : float
        Intended azimuth resolution in meters.
    prf : float
        Pulse repetition frequency in Hz.
    ellipsoid : isce3.core.Ellipsoid, optional
        Ellipsoid describing shape of planet.
    convolution_mode : {"valid", "full", "same"}, optional
        How to handle boundary effects of focusing operation.  For "valid",
        only return regions that will be fully focused.  For "full", return
        regions with any nonzero data.  For "same", return regions that are
        at least halfway focused.
    allowed_azimuth_gap : int, optional
        Amount of time (specified in pulse intervals) allowed between files
        that is still considered contiguous.  If exceeded, the missing data
        will be considered invalid and masked according to `convolution_mode`.

    Returns
    -------
    raw_polygon_lists : list[list[shapely.Polygon]]
        List of valid data regions for each file/observation specified in
        raw radar (x=range, y=time) coordinates (e.g., native Doppler).

    Notes
    -----
    Azimuth times in the RadarBoundingBox and Doppler objects must all be
    relative to the orbit reference epoch.
    """
    if len(raw_bbox_lists) != len(chirp_durations):
        raise ValueError("Need a chirp length for each list of sub-swaths, "
            f"but {len(raw_bbox_lists)=} while {len(chirp_durations)=}")

    if convolution_mode not in {"valid", "full", "same"}:
        raise ValueError(f"Invalid {convolution_mode=}")

    # Copy user input before modifying it.
    raw_bbox_lists = deepcopy(raw_bbox_lists)

    # Handle range convolution effect.  Note that range delay is referenced
    # to the _start_ of the chirp.
    for chirp_duration, bboxes in zip(chirp_durations, raw_bbox_lists):
        chirp_length = chirp_duration * isce3.core.speed_of_light / 2
        for bbox in bboxes:
            if convolution_mode == "full":
                # Extend start
                bbox.first.range -= chirp_length
            elif convolution_mode == "valid":
                # Truncate end
                bbox.last.range -= chirp_length
            elif convolution_mode == "same":
                # Preserve length, allowing partial convolutions on both sides.
                bbox.first.range -= chirp_length / 2
                bbox.last.range -= chirp_length / 2
            else:
                assert False, "Invalid convolution mode"

    # Azimuth aperture length depends on range, so we need to treat the four
    # corners of the bounding box separately.
    class SwathCorners:
        def __init__(self, bbox: RadarBoundingBox):
            self.near_begin = RadarPoint(bbox.first.time, bbox.first.range)
            self.near_end = RadarPoint(bbox.last.time, bbox.first.range)
            self.far_begin = RadarPoint(bbox.first.time, bbox.last.range)
            self.far_end = RadarPoint(bbox.last.time, bbox.last.range)

        def to_polygon(self):
            # CCW order (though CW if viewed with the usual inverted time).
            return shapely.Polygon([(pt.range, pt.time) for pt in
                [self.near_begin, self.far_begin, self.far_end, self.near_end,
                    self.near_begin]])

    raw_corners_lists = [[SwathCorners(bbox) for bbox in bboxes]
        for bboxes in raw_bbox_lists]

    # Handle azimuth convolution effect.  Raw data is in native Doppler geometry
    # by definition, so azimuth time is referenced to _center_ of aperture.
    # Need to handle first and last time boundaries separately since these
    # could be in separate files.  For now assume continuous across file
    # boundaries.
    for corners in raw_corners_lists[0]:
        tmid = (corners.near_begin.time + corners.near_end.time) / 2.0
        cpi_near = get_sar_duration(tmid, corners.near_begin.range,
            orbit, ellipsoid, azres, wavelength)
        cpi_far = get_sar_duration(tmid, corners.far_begin.range,
            orbit, ellipsoid, azres, wavelength)

        if convolution_mode == "full":
            corners.near_begin.time -= cpi_near / 2
            corners.far_begin.time -= cpi_far / 2
        elif convolution_mode == "valid":
            corners.near_begin.time += cpi_near / 2
            corners.far_begin.time += cpi_far / 2
        # else "same" -> no-op

    for corners in raw_corners_lists[-1]:
        tmid = (corners.near_begin.time + corners.near_end.time) / 2.0
        cpi_near = get_sar_duration(tmid, corners.near_begin.range,
            orbit, ellipsoid, azres, wavelength)
        cpi_far = get_sar_duration(tmid, corners.far_begin.range,
            orbit, ellipsoid, azres, wavelength)

        if convolution_mode == "full":
            corners.near_end.time += cpi_near / 2
            corners.far_end.time += cpi_far / 2
        elif convolution_mode == "valid":
            corners.near_end.time -= cpi_near / 2
            corners.far_end.time -= cpi_far / 2
        # else "same" -> no-op

    # One more wrinkle! Mark gaps in azimuth time between files
    nfiles = len(raw_corners_lists)
    for i in range(nfiles - 1):
        # Only bother checking first subswath since they should all have
        # the same time bounds.
        t0 = raw_bbox_lists[i][0].last.time
        t1 = raw_bbox_lists[i + 1][0].first.time
        if (t1 - t0) * prf > allowed_azimuth_gap:
            midpoint = (t1 - t0) / 2
            for corners in raw_corners_lists[i]:
                cpi_near = get_sar_duration(
                    corners.near_end.time, corners.near_end.range,
                    orbit, ellipsoid, azres, wavelength)
                cpi_far = get_sar_duration(
                    corners.far_end.time, corners.far_end.range,
                    orbit, ellipsoid, azres, wavelength)

                if convolution_mode == "full":
                    # Don't extend further than midpoint of gap to avoid
                    # overlapping swaths across files.
                    corners.near_end.time += min(cpi_near / 2, midpoint)
                    corners.far_end.time += min(cpi_far / 2, midpoint)
                elif convolution_mode == "valid":
                    corners.near_end.time -= cpi_near / 2
                    corners.far_end.time -= cpi_far / 2

            for corners in raw_corners_lists[i + 1]:
                cpi_near = get_sar_duration(
                    corners.near_begin.time, corners.near_begin.range,
                    orbit, ellipsoid, azres, wavelength)
                cpi_far = get_sar_duration(
                    corners.far_begin.time, corners.far_begin.range,
                    orbit, ellipsoid, azres, wavelength)

                if convolution_mode == "full":
                    # Don't extend further than midpoint of gap to avoid
                    # overlapping swaths across files.
                    corners.near_begin.time -= min(cpi_near / 2, midpoint)
                    corners.far_begin.time -= min(cpi_far / 2, midpoint)
                elif convolution_mode == "valid":
                    corners.near_begin.time += cpi_near / 2
                    corners.far_begin.time += cpi_far / 2

    return [[corners.to_polygon() for corners in corners_list]
        for corners_list in raw_corners_lists]


def _flatten_nested_lists(list_of_lists):
    return [item for _list in list_of_lists for item in _list]


def transform_polygons_raw2image(*, raw_polygon_lists, orbit, lookside,
                                 native_doppler, wavelength,
                                 image_grid_doppler=isce3.core.LUT2d(),
                                 dem=isce3.geometry.DEMInterpolator(),
                                 max_segment_length=5000.0,
                                 rdr2geo_params=dict(),
                                 geo2rdr_params=dict()):
    """
    Convert valid data polygon radar coordinates from raw data (native Doppler)
    domain to focused image domain (usually zero Doppler for NISAR).

    Parameters
    ----------
    raw_polygon_lists : list[list[shapely.Polygon]]
        List of valid data regions for each file/observation specified in
        raw radar (x=range, y=time) coordinates (e.g., native Doppler).
    orbit : isce3.core.Orbit
        Trajectory of radar antenna phase center.
    lookside : isce3.core.LookSide
        Side that radar observes (Left or Right)
    native_doppler : isce3.core.LUT2d
        Doppler centroid in Hz.  Time should be referenced to the same epoch
        as the orbit.
    wavelength : float
        Wavelength associated with Doppler in meters.
    image_grid_doppler : isce3.core.LUT2d, optional
        Doppler (in Hz) associated with focused image grid geometry.
        Defaults to zero-Doppler (NISAR convention).
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model.  Defaults to 0 m above WGS84 ellipsoid.
    max_segment_length : float, optional
        Length scale over which subswath boundary can be considered linear,
        in meters.
    rdr2geo_params : dict, optional
        Parameters for rdr2geo_bracket
    geo2rdr_params : dict, optional
        Parameters for geo2rdr_bracket

    Returns
    -------
    slc_polygon_lists : list[list[shapely.Polygon]]
        List of valid data regions for each file/observation specified in
        focused radar image (x=range, y=time) coordinates.
    """
    # Get a representative velocity to temporarily convert azimuth time to
    # length units so we can segmentize polygon somewhat uniformly.
    mid_time = np.mean([poly.centroid.y for poly in _flatten_nested_lists(
        raw_polygon_lists)])
    _, vel = orbit.interpolate(mid_time)
    vs = np.linalg.norm(vel)

    # For debugging, it's more intuitive if we remove the time offset.
    t0 = min(poly.bounds[1] for poly in _flatten_nested_lists(
        raw_polygon_lists))

    # Helper function to convert polygon y-units from time (seconds) to space (meters)
    multiply_vel = lambda poly: shapely.affinity.affine_transform(poly, (
        1,  0, 0,
        0, vs, 0,
        0,  0, 1,
        0, -vs * t0, 0))

    # Helper function to convert polygon y-units from space (meters) to time (seconds)
    divide_vel = lambda poly: shapely.affinity.affine_transform(poly, (
        1, 0, 0,
        0, 1 / vs, 0,
        0, 0, 1,
        0, t0, 0))

    slc_polygon_lists = []
    for raw_polygons in raw_polygon_lists:
        slc_polygons = []
        for raw_poly in raw_polygons:
            # Add intermediate points between corners before tranforming
            # geometry.  Spacing specified in length units, so scale
            # (length = velocity * time) and then back.
            raw_poly = multiply_vel(raw_poly).segmentize(max_segment_length)
            raw_poly = divide_vel(raw_poly)

            # Transform from native Doppler to image grid Doppler
            # (always zero for NISAR).
            slc_coords = []
            for (raw_range, raw_time) in zip(*raw_poly.boundary.coords.xy):
                t, r = isce3.geometry.rdr2rdr(raw_time, raw_range, orbit,
                    lookside, native_doppler, wavelength, dem,
                    ellipsoid=dem.ellipsoid,
                    doppler_out=image_grid_doppler,
                    rdr2geo_params=rdr2geo_params,
                    geo2rdr_params=geo2rdr_params)
                slc_coords.append((r, t))
            # Just in case rdr2rdr is not deterministic, force closed polygon.
            slc_coords[-1] = slc_coords[0]
            slc_polygons.append(shapely.Polygon(slc_coords))
        slc_polygon_lists.append(slc_polygons)

    return slc_polygon_lists


def rasterize_subswath_polygons(slc_polygon_lists, slc_grid, threshold=2):
    """
    Convert valid data polygons to [start, stop) indices on the image grid.

    Parameters
    ----------
    slc_polygon_lists : list[list[shapely.Polygon]]
        List of valid data regions for each file/observation specified in
        focused radar image (x=range, y=time) coordinates.
    slc_grid : isce3.product.RadarGridParameters
        Grid for focused image.
    threshold : int
        Number of range samples allowed between adjacent subswaths from
        different observations that will still be considered contiguous and
        merged into a single subswath.  If exceeded, the subswaths will be
        bookkept independently.  (Relevant when reskew is significant.)

    Returns
    -------
    sub_swaths : numpy.ndarray[np.uint32]
        Array of [start, stop) valid data regions, shape = (nswath, npulse, 2)
        where nswath is the number of valid sub-swaths and npulse is the length
        of the focused image grid.
    """

    # The single-file case is straightforward, but with multiple files it's
    # possible to have gaps in varying locations and widths if the PRF or
    # pulse width changes.
    nswaths = max(len(polygons) for polygons in slc_polygon_lists)
    swaths = np.zeros((nswaths, slc_grid.shape[0], 2), dtype=np.uint32)

    # Also the reskew from native to image grid Doppler means the boundaries
    # between files may not be aligned with the grid axes, which may increase
    # the apparent number of sub swaths.  We'll try to be smart and merge
    # them where it's possible and pick the largest where it's not.  Avoid
    # having a bunch of trivial extra subswaths at file boundaries.
    tmp_swaths = np.zeros((nswaths, len(slc_polygon_lists), 2), dtype=np.uint32)

    r0, r1 = slc_grid.slant_ranges[0], slc_grid.slant_ranges[-1]
    dr = slc_grid.range_pixel_spacing

    for itime in range(slc_grid.shape[0]):
        # Scanline corresponding to a row of the radar image grid.
        t = slc_grid.sensing_times[itime]
        line = shapely.LineString([(r0, t), (r1, t)])
        # Initialize to start=end, e.g., all subswaths empty/invalid.
        tmp_swaths[...] = 0

        for iobs, polygon_list in enumerate(slc_polygon_lists):
            # Assume polygons within a single observation are disjoint.
            # Label swath with list index so that "subSwath3" corresponds to
            # the third subswath even if it's the only one with active samples
            # (which can happen with azimuth recording gap + squint).
            for iswath, poly in enumerate(polygon_list):
                geom = line & poly
                if geom.is_empty:
                    # Leave subswath empty.
                    continue
                if geom.geom_type == "LineString":
                    # Intersection is a single line.
                    segment = geom
                else:
                    # Multiple line segments (e.g., due to reskew), so pick
                    # the largest one.
                    segment = max(geom.geoms, key = lambda seg: seg.length)
                r = np.array(sorted(segment.coords.xy[0]))
                tmp_swaths[iswath, iobs, :] = np.round((r - r0) / dr)

        # Initialize with first observation.
        swaths[:, itime, :] = tmp_swaths[:, 0, :]

        # Loop over subsequent observations to 1) check for conflicts and
        # resolve by picking the biggest region labeled "subSwathN" and
        # 2) merge subswaths that are continuous but from different files
        # (e.g., seamless transition in azimuth + reskew).
        for iobs in range(1, len(slc_polygon_lists)):
            for iswath in range(nswaths):
                # previous best estimate and its size
                i0, i1 = swaths[iswath, itime]
                n = i1 - i0
                # estimate for current observation.
                j0, j1 = tmp_swaths[iswath, iobs]

                # Cast to int to prevent unsigned underflow.
                if abs(int(j0) - i1) <= threshold:
                    # Join contiguous, previous first.
                    swaths[iswath, itime] = (i0, j1)
                elif abs(int(i0) - j1) <= threshold:
                    # Join contiguous, current first.
                    swaths[iswath, itime] = (j0, i1)
                elif (j1 - j0) > n:
                    # Conflict for this subswath label, current is bigger.
                    swaths[iswath, itime] = (j0, j1)
                else:
                    # else pass because swath is either invalid or smaller
                    assert (j0 == j1) or ((j1 - j0) <= n)

    return swaths


def get_focused_sub_swaths(raw_bbox_lists, chirp_durations, orbit,
                           native_doppler, azres, grid,
                           dem=isce3.geometry.DEMInterpolator(),
                           image_grid_doppler=isce3.core.LUT2d(),
                           rdr2geo_params=dict(),
                           geo2rdr_params=dict(), max_segment_length=5000,
                           convolution_mode="valid",
                           allowed_azimuth_gap=2, allowed_range_gap=2):
    """
    Determine valid data regions of a focused image, considering transmit
    gaps and gaps between files (in multi-observation processing).

    Parameters
    ----------
    raw_bbox_lists : list[list[RadarBoundingBox]]
        Bounding boxes of all subswaths for each raw data file/observation.
        Azimuth times should be specified in seconds relative to the orbit
        reference epoch.
    chirp_durations : list[float]
        Duration of transmit chirp for each raw data file.
    orbit : isce3.core.Orbit
        Trajectory of radar antenna phase center.
    native_doppler : isce3.core.LUT2d
        Doppler centroid in Hz.  Time should be referenced to the same epoch
        as the orbit.
    azres : float
        Intended azimuth resolution in meters.
    grid : isce3.product.RadarGridParameters
        Grid for focused image.
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model.  Defaults to 0 m above WGS84 ellipsoid.
    image_grid_doppler : isce3.core.LUT2d, optional
        Doppler (in Hz) associated with focused image grid geometry.
        Defaults to zero-Doppler (NISAR convention).
    rdr2geo_params : dict, optional
        Parameters for rdr2geo_bracket
    geo2rdr_params : dict, optional
        Parameters for geo2rdr_bracket
    max_segment_length : float, optional
        Length scale over which subswath boundary can be considered linear,
        in meters.
    convolution_mode : {"valid", "full", "same"}, optional
        How to handle boundary effects of focusing operation.  For "valid",
        only return regions that will be fully focused.  For "full", return
        regions with any nonzero data.  For "same", return regions that are
        at least halfway focused.
    allowed_azimuth_gap : int, optional
        Amount of time (specified in pulse intervals) allowed between files
        that is still considered contiguous.  If exceeded, the missing data
        will be considered invalid and masked according to `convolution_mode`.
    allowed_range_gap : int, optional
        Number of range samples allowed between adjacent subswaths from
        different observations that will still be considered contiguous and
        merged into a single subswath.  If exceeded, the subswaths will be
        bookkept independently.  (Relevant when reskew is significant.)

    Returns
    -------
    sub_swaths : numpy.ndarray[np.uint32]
        Array of [start, stop) valid data regions, shape = (nswath, npulse, 2)
        where nswath is the number of valid sub-swaths and npulse is the length
        of the focused image grid.
    """
    raw_polygons = get_raw_sub_swath_polygons(raw_bbox_lists=raw_bbox_lists,
        chirp_durations=chirp_durations, orbit=orbit,
        wavelength=grid.wavelength, azres=azres, prf=grid.prf,
        ellipsoid=dem.ellipsoid, convolution_mode=convolution_mode,
        allowed_azimuth_gap=allowed_azimuth_gap)

    slc_polygons = transform_polygons_raw2image(raw_polygon_lists=raw_polygons,
        orbit=orbit, lookside=grid.lookside, native_doppler=native_doppler,
        wavelength=grid.wavelength, dem=dem,
        image_grid_doppler=image_grid_doppler,
        max_segment_length=max_segment_length,
        rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params)

    return rasterize_subswath_polygons(slc_polygons, grid,
        threshold=allowed_range_gap)
