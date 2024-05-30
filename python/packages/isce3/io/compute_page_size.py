import math

def compute_page_size(chunk: int):
    """
    Compute the page size in bytes given the chunk memory footprint in bytes

    Parameters
    ----------
    chunk : int
        The chunk memory footprint

    Returns
    -------
    int
        the computed page size in bytes (minimum 4096 bytes)
    """

    # compute the next power of 2 that is strictly larger
    # than the chunk memory footprint
    return max(2 ** math.floor(math.log2(chunk) + 1), 4096)