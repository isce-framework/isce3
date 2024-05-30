import journal

def optimize_chunk_size(unopt_chunk_size, ds_shape):
    """
    Get the optimized chunk size and the chunk cache size in bytes

    Parameters
    ----------
    unopt_chunk_size : tuple(int, int) or tuple(int, int, int)
        The unoptimized chunk shape size
    ds_shape : tuple(int, int) or tuple(int, int, int)
        The shape dataset whose chunk size is to be optimized

    Returns
    -------
        opt_chunk_size: tuple(int, int) or tuple(int, int, int)
            The shape of optimized chunk
    """

    error_channel = journal.error("isce3.io.optimize_chunk_size")

    # Ensure the chunk dimensions matches the dataset dimensions
    if len(unopt_chunk_size) != len(ds_shape):
        err_str = f"the length of the chunk size {unopt_chunk_size} is" +\
        f" not equal to length of the dataset shape size {ds_shape}"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Ensure only 2d and 3d datasets are passed in
    ds_ndims = len(ds_shape)
    if ds_ndims not in [2, 3]:
        err_str = f"Dataset has {ds_ndims} dimensions." +\
        "Only 2D and 3D dimensions allowed"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Compute the chunk size for the datacube
    # if the the dataset shape is smaller than the chunk size
    # the chunk size will be the shape of the dataset
    opt_min_chunk_size = list(map(min, zip(unopt_chunk_size, ds_shape)))

    # The chunk size of the first dimension of the 3D dataset will always be 1
    if ds_ndims == 3:
        opt_min_chunk_size[0] = 1

    return tuple(opt_min_chunk_size)