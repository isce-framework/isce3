import copy
import h5py
import numpy as np


class HDF5OptimizedReader(h5py.File):
    """
    The HDF5 optimizer reader class inheriting from h5py.File
    to avoid passing h5py.File parameter
    """

    def __init__(self,name, **kwds):
        """
        Constructor of the HDF5 optimizer reader inheriting
        from h5py.File to avoid passing h5py.File parameter.

        Parameters
        ----------
        name : str
            HDF5 file name
        num_rows : int, optional
            Minimal number of rows you want to be able to cache (default: 2048)
        kwds
            Keyword arguments forwarded to h5py.File
        """

        # To avoid the change of the kwds in-place
        new_kwds = copy.deepcopy(kwds)
        hdf5_file = name
        num_rows = new_kwds.pop('num_rows', 2048)

        # The minimum chunk cache size is set to 1 Mb
        largest_chunk_cache_size = 1024 ** 2

        # Get the largest chunk cache size
        def _get_largest_chunk_cache_size(ds_name, ds):
            """
            Get the largest chunk cache size

            Parameters
            ----------
            ds_name : str
                Dataset name
            ds : h5py.Dataset
                h5py Dataset object
            """

            # nonlocal so largest_chunk_cache_size declared
            # above can be altered when this helper function
            # is iteratively applied to h5py datasets below
            nonlocal largest_chunk_cache_size

            if isinstance(ds, h5py.Dataset):
                ds_ndims = len(ds.shape)
                if ds_ndims in [2,3] and ds.chunks is not None:
                    i_width_dim = ds_ndims - 1
                    i_length_dim = ds_ndims - 2
                    # Ensure that the number of chunk blocks is large
                    # enough to cover the width of the image
                    num_of_blocks = int(
                        float(ds.shape[i_width_dim] +
                              ds.chunks[i_width_dim] - 1.0)/
                         ds.chunks[i_width_dim])

                    chunk_cache_size = \
                        num_of_blocks * np.prod(ds.chunks)\
                            * ds.dtype.itemsize

                    chunk_cache_size *= int(
                        float(num_rows + ds.chunks[i_length_dim] - 1.0)/
                         ds.chunks[i_length_dim])

                    largest_chunk_cache_size = \
                        max(largest_chunk_cache_size, chunk_cache_size)

        with h5py.File(hdf5_file, **new_kwds) as h5:
            h5.visititems(_get_largest_chunk_cache_size)

        new_kwds['rdcc_nbytes'] = largest_chunk_cache_size

        # Initialize the h5py File object
        super().__init__(hdf5_file,**new_kwds)