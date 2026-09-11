from pathlib import Path

import numpy as np
from osgeo import gdal


def _solve_offset_block(A, L):
    """
    Solve:

        L_ij = u_j - u_i

    for one block of pixels.

    Parameters
    ----------
    A : (n_pairs, n_dates - 1)
        Network design matrix with reference date removed.

    L : (n_pairs, n_pixels)
        Pairwise offsets.

    Returns
    -------
    U : (n_dates - 1, n_pixels)
        Per-date offsets relative to reference.

    Notes
    -----
    Handles NaNs. If all pairs are valid, uses one matrix
    multiplication. For pixels containing NaNs, uses the
    available pair network if it remains full rank.
    """

    n_unknown = A.shape[1]
    n_pix = L.shape[1]

    U = np.full(
        (n_unknown, n_pix),
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------
    # Fast path: all pairs valid
    # --------------------------------------------------

    full_valid = np.all(np.isfinite(L), axis=0)

    if np.any(full_valid):
        Ainv = np.linalg.pinv(A)

        U[:, full_valid] = (
            Ainv @ L[:, full_valid]
        )

    # --------------------------------------------------
    # Pixels with missing pair observations
    # --------------------------------------------------

    bad_pixels = np.where(~full_valid)[0]

    # Cache pseudoinverses for repeated validity patterns
    pinv_cache = {}

    for k in bad_pixels:

        valid_pairs = np.isfinite(L[:, k])

        key = valid_pairs.tobytes()

        if key not in pinv_cache:

            Av = A[valid_pairs, :]

            # Need a connected/full-rank network
            if (
                Av.shape[0] >= n_unknown
                and np.linalg.matrix_rank(Av) == n_unknown
            ):
                pinv_cache[key] = np.linalg.pinv(Av)
            else:
                pinv_cache[key] = None

        Ainv = pinv_cache[key]

        if Ainv is not None:
            U[:, k] = (
                Ainv
                @ L[valid_pairs, k]
            )

    return U


def invert_offsets_isce3(
    pair_offsets,
    output_dir,
    reference_date,
    block_rows=128,
):
    """
    Network-invert pairwise ISCE3 dense offsets.

    Parameters
    ----------
    pair_offsets : dict
        Dictionary mapping:

            (reference_date, secondary_date) -> dense_offsets path

        Example:

        {
            ("20260617", "20260629"):
                "/.../20260617_20260629/dense_offsets",

            ("20260629", "20260711"):
                "/.../20260629_20260711/dense_offsets",

            ("20260617", "20260711"):
                "/.../20260617_20260711/dense_offsets",
        }

        Each raster must contain:

            band 1 = azimuth offset
            band 2 = range offset

        All rasters must have identical dimensions AND
        represent identical Ampcor window locations.

    output_dir : str or Path
        Output directory.

    reference_date : str
        Stack reference date. Its inverted residual offset
        is defined as zero.

    block_rows : int
        Number of offset-grid rows processed at once.

    Returns
    -------
    dict
        Mapping:

            date -> output dense_offsets path

    Output
    ------
    output_dir/
        20260617/dense_offsets
        20260629/dense_offsets
        20260711/dense_offsets
        ...

    Each output is a 2-band ENVI Float32 raster:

        band 1 = azimuth residual relative to reference
        band 2 = range residual relative to reference
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Normalize inputs
    pair_offsets = {
        (str(d1), str(d2)): Path(path)
        for (d1, d2), path in pair_offsets.items()
    }

    pairs = list(pair_offsets.keys())

    if len(pairs) == 0:
        raise ValueError(
            "No pair offset rasters supplied."
        )

    # --------------------------------------------------
    # Build date list
    # --------------------------------------------------

    dates = sorted(
        {
            date
            for pair in pairs
            for date in pair
        }
    )

    reference_date = str(reference_date)

    if reference_date not in dates:
        raise ValueError(
            f"Reference date {reference_date} "
            f"is not in network: {dates}"
        )

    print("Dates:")
    for date in dates:
        marker = " <-- reference" if date == reference_date else ""
        print(f"    {date}{marker}")

    print()
    print("Pairs:")
    for pair in pairs:
        print(
            f"    {pair[0]} -> {pair[1]}"
        )

    # --------------------------------------------------
    # Construct full incidence matrix
    #
    # pair measurement:
    #
    #     offset_ij = offset_j - offset_i
    #
    # --------------------------------------------------

    n_pairs = len(pairs)
    n_dates = len(dates)

    date_index = {
        date: i
        for i, date in enumerate(dates)
    }

    A_full = np.zeros(
        (n_pairs, n_dates),
        dtype=np.float64,
    )

    for k, (d1, d2) in enumerate(pairs):

        A_full[
            k,
            date_index[d1]
        ] = -1.0

        A_full[
            k,
            date_index[d2]
        ] = +1.0

    # --------------------------------------------------
    # Fix reference date = 0
    # --------------------------------------------------

    unknown_dates = [
        date
        for date in dates
        if date != reference_date
    ]

    unknown_columns = [
        date_index[date]
        for date in unknown_dates
    ]

    A = A_full[:, unknown_columns]

    print()
    print("Design matrix:")
    print(A)

    # --------------------------------------------------
    # Check global network connectivity
    # --------------------------------------------------

    rank = np.linalg.matrix_rank(A)

    if rank != len(unknown_dates):
        raise ValueError(
            "Offset network is not fully connected to the "
            f"reference. rank={rank}, "
            f"required={len(unknown_dates)}"
        )

    print(
        f"Network rank: {rank}/"
        f"{len(unknown_dates)}"
    )

    # --------------------------------------------------
    # Open input rasters
    # --------------------------------------------------

    input_ds = []

    width = None
    length = None

    for pair in pairs:

        path = pair_offsets[pair]

        ds = gdal.Open(
            str(path),
            gdal.GA_ReadOnly,
        )

        if ds is None:
            raise RuntimeError(
                f"Could not open {path}"
            )

        if ds.RasterCount < 2:
            raise ValueError(
                f"{path} has {ds.RasterCount} bands; "
                "expected at least 2."
            )

        if width is None:
            width = ds.RasterXSize
            length = ds.RasterYSize

        elif (
            ds.RasterXSize != width
            or ds.RasterYSize != length
        ):
            raise ValueError(
                "All pair offsets must have identical "
                "dimensions.\n"
                f"Expected: {length} x {width}\n"
                f"{path}: "
                f"{ds.RasterYSize} x "
                f"{ds.RasterXSize}"
            )

        input_ds.append(ds)

    print()
    print(
        f"Offset grid: {length} x {width}"
    )

    # --------------------------------------------------
    # Create output rasters
    # --------------------------------------------------

    driver = gdal.GetDriverByName("ENVI")

    output_ds = {}
    output_paths = {}

    for date in dates:

        date_dir = output_dir / date
        date_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = date_dir / "dense_offsets"

        ds = driver.Create(
            str(path),
            width,
            length,
            2,
            gdal.GDT_Float32,
            options=[
                "INTERLEAVE=BIP",
            ],
        )

        ds.GetRasterBand(1).SetDescription(
            "azimuth_offset"
        )

        ds.GetRasterBand(2).SetDescription(
            "range_offset"
        )

        output_ds[date] = ds
        output_paths[date] = path

    # --------------------------------------------------
    # Process by row blocks
    # --------------------------------------------------

    for y0 in range(
        0,
        length,
        block_rows,
    ):

        nrows = min(
            block_rows,
            length - y0,
        )

        print(
            f"Rows {y0}:{y0 + nrows} "
            f"of {length}"
        )

        # Pair stacks:
        #
        #     (n_pairs, nrows, width)
        #

        az_pairs = np.empty(
            (n_pairs, nrows, width),
            dtype=np.float64,
        )

        rg_pairs = np.empty_like(
            az_pairs
        )

        for k, ds in enumerate(input_ds):

            az_pairs[k] = (
                ds.GetRasterBand(1)
                .ReadAsArray(
                    0,
                    y0,
                    width,
                    nrows,
                )
            )

            rg_pairs[k] = (
                ds.GetRasterBand(2)
                .ReadAsArray(
                    0,
                    y0,
                    width,
                    nrows,
                )
            )

        # Flatten spatial dimensions
        #
        # (n_pairs, n_pixels)
        #

        az_L = az_pairs.reshape(
            n_pairs,
            -1,
        )

        rg_L = rg_pairs.reshape(
            n_pairs,
            -1,
        )

        # ----------------------------------------------
        # Network inversion
        # ----------------------------------------------

        az_U = _solve_offset_block(
            A,
            az_L,
        )

        rg_U = _solve_offset_block(
            A,
            rg_L,
        )

        # ----------------------------------------------
        # Reference = zero
        # ----------------------------------------------

        ref_az = np.zeros(
            (nrows, width),
            dtype=np.float32,
        )

        ref_rg = np.zeros_like(
            ref_az
        )

        output_ds[
            reference_date
        ].GetRasterBand(1).WriteArray(
            ref_az,
            0,
            y0,
        )

        output_ds[
            reference_date
        ].GetRasterBand(2).WriteArray(
            ref_rg,
            0,
            y0,
        )

        # ----------------------------------------------
        # Other dates
        # ----------------------------------------------

        for i, date in enumerate(
            unknown_dates
        ):

            az = az_U[i].reshape(
                nrows,
                width,
            ).astype(np.float32)

            rg = rg_U[i].reshape(
                nrows,
                width,
            ).astype(np.float32)

            output_ds[
                date
            ].GetRasterBand(1).WriteArray(
                az,
                0,
                y0,
            )

            output_ds[
                date
            ].GetRasterBand(2).WriteArray(
                rg,
                0,
                y0,
            )

    # --------------------------------------------------
    # Close datasets
    # --------------------------------------------------

    for ds in output_ds.values():
        band1 = ds.GetRasterBand(1)
        band2 = ds.GetRasterBand(2)
        band1.FlushCache()
        band2.FlushCache()
        ds.FlushCache()
    output_ds.clear()
    input_ds.clear()

    print()
    print("Offset inversion complete.")

    for date in dates:
        print(
            f"{date}: {output_paths[date]}"
        )

    return output_paths


def _add_one_offset(
    offset1,
    offset2,
    output,
    block_rows=1024,
    invalid_threshold=1.0e5,
    invalid_value=-1.0e6,
):
    """
    Add two single-band GDAL-readable offset rasters and
    write a Float64 ISCE raster.
    """

    offset1 = Path(offset1)
    offset2 = Path(offset2)
    output = Path(output)

    ds1 = gdal.Open(
        str(offset1),
        gdal.GA_ReadOnly,
    )

    if ds1 is None:
        raise RuntimeError(
            f"Could not open {offset1}"
        )

    ds2 = gdal.Open(
        str(offset2),
        gdal.GA_ReadOnly,
    )

    if ds2 is None:
        ds1 = None

        raise RuntimeError(
            f"Could not open {offset2}"
        )

    if ds1.RasterCount != 1:
        raise ValueError(
            f"{offset1} has {ds1.RasterCount} bands; "
            "expected 1."
        )

    if ds2.RasterCount != 1:
        raise ValueError(
            f"{offset2} has {ds2.RasterCount} bands; "
            "expected 1."
        )

    width1 = ds1.RasterXSize
    length1 = ds1.RasterYSize

    width2 = ds2.RasterXSize
    length2 = ds2.RasterYSize

    if (
        width1 != width2
        or length1 != length2
    ):
        raise ValueError(
            "Offset dimensions do not match:\n"
            f"  {offset1}: {length1} x {width1}\n"
            f"  {offset2}: {length2} x {width2}"
        )

    width = width1
    length = length1

    band1 = ds1.GetRasterBand(1)
    band2 = ds2.GetRasterBand(1)

    nodata1 = band1.GetNoDataValue()
    nodata2 = band2.GetNoDataValue()

    dtype1 = gdal.GetDataTypeName(
        band1.DataType
    )

    dtype2 = gdal.GetDataTypeName(
        band2.DataType
    )

    print()
    print("Adding:")
    print("  input 1:", offset1)
    print("           dtype:", dtype1)
    print("  input 2:", offset2)
    print("           dtype:", dtype2)
    print("  output :", output)
    print("           dtype: Float64")
    print(
        "  shape  :",
        f"{length} x {width}",
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    driver = gdal.GetDriverByName(
        "ISCE"
    )

    if driver is None:
        raise RuntimeError(
            "GDAL ISCE driver is not available."
        )

    out_ds = driver.Create(
        str(output),
        width,
        length,
        1,
        gdal.GDT_Float64,
    )

    if out_ds is None:
        raise RuntimeError(
            f"Could not create {output}"
        )

    out_band = out_ds.GetRasterBand(1)

    for y0 in range(
        0,
        length,
        block_rows,
    ):

        nrows = min(
            block_rows,
            length - y0,
        )

        y1 = y0 + nrows

        print(
            f"    rows {y0}:{y1} / {length}"
        )

        a = band1.ReadAsArray(
            0,
            y0,
            width,
            nrows,
        ).astype(
            np.float64,
            copy=False,
        )

        b = band2.ReadAsArray(
            0,
            y0,
            width,
            nrows,
        ).astype(
            np.float64,
            copy=False,
        )

        invalid = (
            ~np.isfinite(a)
            | ~np.isfinite(b)
        )

        if nodata1 is not None:
            invalid |= (
                a == nodata1
            )

        if nodata2 is not None:
            invalid |= (
                b == nodata2
            )

        if invalid_threshold is not None:

            invalid |= (
                np.abs(a)
                >= invalid_threshold
            )

            invalid |= (
                np.abs(b)
                >= invalid_threshold
            )

        result = a + b

        result[invalid] = (
            invalid_value
        )

        out_band.WriteArray(
            result,
            0,
            y0,
        )

    out_band.FlushCache()
    out_ds.FlushCache()

    ds1 = None
    ds2 = None
    out_ds = None


def add_offset_folders(
    folder1,
    folder2,
    outfolder,
    block_rows=1024,
    invalid_threshold=1.0e5,
    invalid_value=-1.0e6,
):
    """
    Add azimuth.off and range.off from two folders.

    Expected input:

        folder1/
            azimuth.off
            range.off

        folder2/
            azimuth.off
            range.off

    Output:

        outfolder/
            azimuth.off
            azimuth.off.xml
            range.off
            range.off.xml

    Operation:

        out/azimuth.off =
            folder1/azimuth.off
            +
            folder2/azimuth.off

        out/range.off =
            folder1/range.off
            +
            folder2/range.off
    """

    folder1 = Path(folder1)
    folder2 = Path(folder2)
    outfolder = Path(outfolder)

    outfolder.mkdir(
        parents=True,
        exist_ok=True,
    )

    az1 = folder1 / "azimuth.off"
    rg1 = folder1 / "range.off"

    az2 = folder2 / "azimuth.off"
    rg2 = folder2 / "range.off"

    out_az = outfolder / "azimuth.off"
    out_rg = outfolder / "range.off"

    required = [
        az1,
        rg1,
        az2,
        rg2,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required offset files:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    print("Folder 1:", folder1)
    print("Folder 2:", folder2)
    print("Output  :", outfolder)

    # --------------------------------------------
    # Azimuth
    # --------------------------------------------

    _add_one_offset(
        offset1=az1,
        offset2=az2,
        output=out_az,
        block_rows=block_rows,
        invalid_threshold=invalid_threshold,
        invalid_value=invalid_value,
    )

    # --------------------------------------------
    # Range
    # --------------------------------------------

    _add_one_offset(
        offset1=rg1,
        offset2=rg2,
        output=out_rg,
        block_rows=block_rows,
        invalid_threshold=invalid_threshold,
        invalid_value=invalid_value,
    )

    print()
    print("Done.")
    print("Azimuth:", out_az)
    print("Range  :", out_rg)

    return {
        "azimuth": out_az,
        "range": out_rg,
    }
