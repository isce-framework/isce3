from __future__ import annotations

from typing import TypedDict, TypeVar

import numpy as np

import isce3

T = TypeVar("T")


class XYDict(TypedDict):
    x: float | None
    y: float | None


def get_output_geo_grid(
    top_left: XYDict,
    bottom_right: XYDict,
    posting: XYDict,
    epsg: int | None,
    dem_raster: isce3.io.Raster,
) -> isce3.product.GeoGridParameters:
    """
    Determine the geo grid of the Static Layers granule from the runconfig parameters.

    If `epsg` is provided and differs from the EPSG code of the digital elevation model
    (DEM) raster coordinate reference system, then the remaining parameters of the
    output geo grid (`top_left`, `bottom_right`, and `posting`) must be explicitly
    specified. Otherwise, each parameter is inferred from the DEM raster if unspecified.

    Parameters
    ----------
    top_left : dict
        The X and Y coordinates of the upper-left corner of the geo grid bounding box,
        in the native units of the coordinate reference system (CRS) defined by the
        `epsg` argument. A dict containing 'x' and 'y' keys. If `epsg` is not `None` and
        differs from the DEM raster EPSG code, both parameters must be specified.
        Otherwise, if either value in the dict is `None`, it is inferred from the
        bounding box of `dem_raster`.
    bottom_right : dict
        The X and Y coordinates of the lower-right corner of the geo grid bounding box,
        in the native units of the CRS defined by the `epsg` argument. A dict containing
        'x' and 'y' keys. If `epsg` is not `None` and differs from the DEM raster EPSG
        code, both parameters must be specified. Otherwise, if either value in the dict
        is `None`, it is inferred from the bounding box of `dem_raster`.
    posting : dict
        The X and Y pixel spacing of the output grid, in the native units of the CRS
        defined by the `epsg` argument. A dict containing 'x' and 'y' keys. If `epsg` is
        not `None` and differs from the DEM raster EPSG code, both parameters must be
        specified.  Otherwise, if either value in the dict is `None`, it is inferred
        from the pixel spacing of `dem_raster`. The spacing should always be
        positive-valued (the output grid always has north-up, west-left orientation).
    epsg : int or None
        The EPSG code of the CRS of the output geocoded grid. If None, the EPSG code of
        `dem_raster` is used.
    dem_raster : isce3.io.Raster
        The input DEM raster dataset.

    Returns
    -------
    isce3.product.GeoGridParameters
        The geocoded coordinate grid on which the raster data of the Static Layers
        product granule should be sampled.
    """

    # Returns `param` if not None; otherwise `default`.
    def get_option(param: T | None, *, default: T) -> T:
        return param if (param is not None) else default

    # EPSG code of the output geo grid.
    dem_epsg = dem_raster.get_epsg()
    epsg = get_option(epsg, default=dem_epsg)

    # If the output grid EPSG matches the DEM EPSG, then any parameters of the output
    # grid may be inferred from the input DEM if not explicitly specified by the user.
    # Otherwise, if the output grid EPSG differs from the DEM EPSG, each geo grid
    # parameter must be explicitly specified.
    if epsg == dem_epsg:
        # X and Y coordinate of the upper-left corner pixel of the output geo grid.
        xmin = get_option(top_left["x"], default=dem_raster.x0)
        ymax = get_option(top_left["y"], default=dem_raster.y0)

        # X and Y spacing of the output geo grid.
        dx = get_option(posting["x"], default=dem_raster.dx)
        dy = -posting["y"] if (posting["y"] is not None) else dem_raster.dy

        # X and Y coordinate of the lower-right corner pixel of the DEM.
        dem_x_lr = dem_raster.x0 + dem_raster.dx * dem_raster.width
        dem_y_lr = dem_raster.y0 + dem_raster.dy * dem_raster.length

        # X and Y coordinate of the lower-right corner pixel of the output geo grid.
        xmax = get_option(bottom_right["x"], default=dem_x_lr)
        ymin = get_option(bottom_right["y"], default=dem_y_lr)
    else:
        xmin = top_left["x"]
        xmax = bottom_right["x"]
        ymin = bottom_right["y"]
        ymax = top_left["y"]
        dx = posting["x"]
        dy = -posting["y"]

    # X spacing must be positive and Y spacing must be negative.
    if not (dx > 0.0) or not (dy < 0.0):
        raise ValueError(
            f"{dx=} must be positive-valued and {dy=} must be negative-valued"
        )

    # Ensure that the bounding box is valid.
    if not (xmax > xmin) or not (ymax > ymin):
        raise ValueError(
            f"invalid bounding box: {xmax=} must be > {xmin=} and {ymax=} must be >"
            f" {ymin=}"
        )

    # Divide `num` by `den` and round the result to the nearest integer.
    def round_divide(num: float, den: float) -> int:
        return int(np.round(num / den))

    # Length and width of the output geo grid.
    # XXX: It seems more sensible to always round up rather than rounding to the nearest
    # integer to ensure the geo grid covers the entire requested bounding box, but the
    # other L2 SAS workflows do it this way. We should ensure this works the same way so
    # that the grids for any particular frame are consistent. See
    # https://github.com/isce-framework/isce3/issues/155.
    length = round_divide(ymin - ymax, dy)
    width = round_divide(xmax - xmin, dx)

    return isce3.product.GeoGridParameters(
        start_x=xmin,
        start_y=ymax,
        spacing_x=dx,
        spacing_y=dy,
        width=width,
        length=length,
        epsg=epsg,
    )
