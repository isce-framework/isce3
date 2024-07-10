# Class containing units to be allocated in the product
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Units:
    """
    Convenience dataclass for storing units in InSAR products
    """
    meter: np.bytes_ = np.bytes_('meters')
    meter2: np.bytes_ = np.bytes_('meters^2')
    second: np.bytes_ = np.bytes_('seconds')
    unitless: np.bytes_ = np.bytes_('1')
    dn: np.bytes_ = np.bytes_('DN')
    radian: np.bytes_ = np.bytes_('radians')
    hertz: np.bytes_ = np.bytes_('hertz')
    rad_per_second: np.bytes_ = np.bytes_('radians / second')
    meter_per_second: np.bytes_ = np.bytes_('meters / second')

