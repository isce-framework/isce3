# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import h5py
import pyre

from . import GenericSingleSourceL2Product

PRODUCT = 'GCOV'


class GCOV(GenericSingleSourceL2Product, family='nisar.productreader.gcov'):
    """
    Class for parsing NISAR GCOV products into ISCE3 structures.
    """

    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'

    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    def covarianceTermsByFreq(self, frequency: str) -> list[str]:
        """
        Returns the list of GCOV covariance terms for the given frequency

        Parameters
        ----------
        frequency: str
            The frequency letter associated with the data array.

        Returns
        -------
        list_of_covariance_terms: list(str)
            The list of GCOV covariance terms for given frequency.
        """

        from nisar.h5 import bytestring, extractWithIterator

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            root = f"{self.GridPath}/frequency{frequency}"
            cov_terms_list = extractWithIterator(
                fid[root],
                'listOfCovarianceTerms',
                bytestring,
                msg=(
                    'Could not determine list of covariance terms for'
                    f' frequency {frequency}'
                )
            )

            list_of_covariance_terms = [c.upper() for c in cov_terms_list]

            return list_of_covariance_terms

    @property
    def covarianceTerms(self) -> dict[str, list[str]]:
        """
        Returns a dictionary of GCOV covariance terms (dictionary values)
        for each available frequency (dictionary keys)

        Returns
        -------
        dict_of_covariance_terms: dict
            A dictionary of GCOV covariance terms (dictionary values)
            for each available frequency (dictionary keys)
        """
        dict_of_covariance_terms = {
            freq: self.covarianceTermsByFreq(freq) for freq in self.frequencies
        }

        return dict_of_covariance_terms
