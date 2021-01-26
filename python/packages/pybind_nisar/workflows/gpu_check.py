import journal

def cuda_device_valid(gpu_id: int) -> bool:
    """
    Validate that the requested CUDA device is supported.

    Parameters
    ----------
    gpu_id : int
        CUDA device index

    Returns
    -------
    logical
        Whether given device is supported
    """
    from pybind_isce3.cuda.core import Device, min_compute_capability
    device = Device(gpu_id)
    return device.compute_capability >= min_compute_capability()


def use_gpu(gpu_requested: bool, gpu_id: int) -> bool:
    """Validate the specified GPU processing configuration options.

    Parameters
    ----------
    gpu_requested : logical
        Whether to use GPU processing. If None, enable GPU processing if available.
    gpu_id : int
        CUDA device index

    Returns
    -------
    logical
        Whether to use GPU processing

    Raises
    ------
    ValueError
        If GPU processing was requested but not available or if an invalid CUDA
        device was requested
    """
    import pybind_isce3

    # Check if CUDA support is enabled.
    cuda_available = lambda : hasattr(pybind_isce3, "cuda")

    # If unspecified, use GPU processing if supported. Otherwise, fall back to
    # CPU processing.
    if gpu_requested is None:
        return cuda_available() and cuda_device_valid(gpu_id)

    # If GPU processing was requested, raise an error if CUDA support is not
    # available or if the specified device is not valid.
    if gpu_requested:
        error_channel = journal.error("gpu_check.use_gpu")
        if not cuda_available():
            # XXX logging an error does not halt execution
            errmsg = "GPU processing was requested but not available"
            error_channel.log(errmsg)
            raise ValueError(errmsg)

        if not cuda_device_valid(gpu_id):
            errmsg = "The requested CUDA device has insufficient compute " \
                     "capability"
            error_channel.log(errmsg)
            raise ValueError(errmsg)

        return True

    # GPU processing was not requested.
    return False
