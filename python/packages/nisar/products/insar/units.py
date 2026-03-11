# Class containing units to be allocated in the product
from dataclasses import dataclass


@dataclass(frozen=True)
class Units:
    """
    Convenience dataclass for storing units in InSAR products
    """
    meter: str = 'meters'
    meter2: str = 'meters^2'
    second: str = 'seconds'
    unitless: str = '1'
    dn: str = 'DN'
    radian: str = 'radians'
    hertz: str = 'hertz'
    rad_per_second: str = 'radians / second'
    meter_per_second: str = 'meters / second'
    days: str = 'days'

