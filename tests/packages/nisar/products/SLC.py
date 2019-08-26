#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import numpy as np

import nisar

n_row = 3
n_col = 4

def slcSumCheck(slc_sum):
    '''
    check if complex array fits following formula:
    i:[0,num_rows)
    j:[0,num_columns)
    value of each element for all freq+pol SLC combos is:
    k = i*num_columns + j; slc[i,j] = complex(k,k)
    sum of any 2 datasets is:
    slc_sum[i,j] = 2*complex(k,k)
    '''
    for i_row in range(n_row):
        for i_col in range(n_col):
            k = 2*(i_row*n_col + i_col)
            if slc_sum[i_row,i_col] != k + k * 1j:
                return False
    return True


def test_slc_swmr():
    '''
    open multiple SLC datasets from same hdf5
    add different open SLC datasets
    check if sums are correct
    '''
    # open SLC from hdf5
    f = 'swmr_test.h5'
    slc_obj = nisar.products.readers.SLC(hdf5file=f)

    # use A+VV SLC to test other freq+pol SLC with swmr mode 
    ds0 = slc_obj.getSlcDataset('A', 'VV')
    assert(ds0.shape == (n_row, n_col))

    # open each freq+pol dataset, sum, and check values
    # define SLC key parameters
    freqs = ['A', 'B']
    pols = ['HH', 'VV', 'HV', 'VH']
    for freq in freqs:
        for pol in pols:
            ds1 = slc_obj.getSlcDataset(freq, pol)
            ds_sum = ds0[:] + ds1[:]
            assert(slcSumCheck(ds_sum))
    return


if __name__ == '__main__':
    test_slc_swmr()

# end of file
