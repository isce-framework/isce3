from typing import Final
from textwrap import dedent

def unwrap(text):
    """Replace newlines with spaces unless preceded by a hyphen.
    """
    return text.replace("-\n", "").replace("\n", " ")

bounding_polygon: Final[str] = unwrap(dedent("""\
    OGR compatible WKT representing the bounding polygon of the image.
    Horizontal coordinates are WGS84 longitude followed by latitude (both in
    degrees), and the vertical coordinate is the height above the WGS84
    ellipsoid in meters. The first point corresponds to the start-time,
    near-range radar coordinate, and the perimeter is traversed in counter-
    clockwise order on the map. This means the traversal order in radar
    coordinates differs for left-looking and right-looking sensors.
    The polygon includes the four corners of the radar grid, with
    equal numbers of points distributed evenly in radar coordinates along
    each edge"""))
