#!/usr/bin/env python3

import iscetest
from nisar.products.readers import open_product, RSLC, GCOV
from nisar.products.readers.Raw import Raw


def test_generic_product():
    '''
    Test that the open_product function returns the correct kind of product,
    and that the generation of product readers works more generally.
    '''
    # Test creation of an L0B RRSD reader
    _check_product("REE_L0B_out17.h5", Raw, 'RRSD')

    # Test creation several L1 RSLC readers
    _check_product("envisat.h5", RSLC, 'RSLC')
    _check_product("Greenland.h5", RSLC, 'RSLC')
    _check_product("winnipeg.h5", RSLC, 'RSLC')

    # Test creation of a GCOV reader
    gcov_product = _check_product("nisar_129_gcov_crop.h5", GCOV, 'GCOV')
    assert gcov_product.covarianceTermsByFreq('A') == ['HHHH']
    assert gcov_product.covarianceTermsByFreq('B') == ['HHHH', 'HVHV', 'VVVV']
    assert gcov_product.covarianceTerms == {'A': ['HHHH'],
                                            'B': ['HHHH', 'HVHV', 'VVVV']}


def _check_product(data_path, product_class, product_type_name) -> None:
    """
    Check that the product returned by open_product is what is expected.

    Parameters
    ----------
    data_path : str
        The path of the data to be checked relative to the `isce3test.data`
        directory.
    product_class : Product reader type
        The type of product reader that is expected.
    product_type_name : str
        The expected productType name on the product reader object.
    """
    nisar_product_obj = \
        open_product(iscetest.data + data_path)
    assert isinstance(nisar_product_obj, product_class)
    assert nisar_product_obj.productType == product_type_name

    return nisar_product_obj
