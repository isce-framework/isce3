# -*- coding: utf-8 -*-
import h5py
import numpy as np
import pyre
from isce3.core import speed_of_light

from .. import GenericSingleSourceL2Product
from .SLCBase import SLCBase

PRODUCT = 'GSLC'


class GSLC(SLCBase, GenericSingleSourceL2Product, family='nisar.productreader.gslc'):
    """
    Class for parsing NISAR GSLC products into ISCE3 structures.
    """

    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'

    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    @property
    def topographicFlatteningApplied(self) -> bool:
        """
        True if this product has been topographically flattened, False otherwise.
        """
        boolPath = (
            f"{self.ProcessingInformationPath}/parameters/topographicFlatteningApplied"
        )

        # open H5 with swmr mode enabled
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            # get dataset
            dataset: h5py.Dataset = fid[boolPath]

            value = dataset[()].decode()

        if value.title() not in ["True", "False"]:
            raise ValueError(
                f"topographicFlatteningApplied value in GSLC product was \"{value}\"; "
                'must be "True" or "False".'
            )

        return value.title() == "True"
