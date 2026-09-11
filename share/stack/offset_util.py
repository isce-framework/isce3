from pathlib import Path
import isce3
from osgeo import gdal
from nisar.workflows.dense_offsets import (
    set_optional_attributes,
    create_empty_dataset,
)
from nisar.workflows.helpers import (get_cfg_freq_pols,
                                     get_offset_radar_grid)
import h5py
import numpy as np
from nisar.products.readers import SLC
from nisar.products.insar.product_paths import RIFGGroupsPaths

def create_minimal_rifg_for_rubbersheet(
    out_h5,
    cfg,
    overwrite=True,
):
    """
    Create only the HDF5 datasets required by nisar.workflows.rubbersheet.

    Parameters
    ----------
    out_h5 : str or Path
        Output minimal RIFG HDF5.
    cfg : dict
        cfg = ref_config["runconfig"]["groups"]
    overwrite : bool
        Replace existing file.
    """

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    if out_h5.exists():
        if overwrite:
            out_h5.unlink()
        else:
            raise FileExistsError(out_h5)

    ref_file = cfg["input_file_group"]["reference_rslc_file"]
    ref_slc = SLC(hdf5file=ref_file)

    with h5py.File(out_h5, "w", libver="latest") as f:

        for freq, _, pol_list in get_cfg_freq_pols(cfg):

            # Reference radar grid
            ref_grid = ref_slc.getRadarGrid(freq)

            # EXACT NISAR pixel-offset grid derived from dense-offset config
            off_grid = get_offset_radar_grid(cfg, ref_grid)

            off_length = off_grid.length
            off_width = off_grid.width

            # Pixel-offset coordinates.
            # These are at the centers of the Ampcor matching windows.
            slant_range = (
                off_grid.starting_range
                + np.arange(off_width, dtype=np.float64)
                * off_grid.range_pixel_spacing
            )

            zero_doppler_time = (
                off_grid.sensing_start
                + np.arange(off_length, dtype=np.float64)
                / off_grid.prf
            )

            base = (
                f"{RIFGGroupsPaths().SwathsPath}"
                f"/frequency{freq}/pixelOffsets"
            )

            g = f.require_group(base)

            # ----------------------------------------------------------
            # Required coordinate vectors
            # ----------------------------------------------------------

            ds = g.create_dataset(
                "slantRange",
                data=slant_range,
                dtype=np.float64,
            )
            ds.attrs["units"] = np.bytes_("meters")

            ds = g.create_dataset(
                "zeroDopplerTime",
                data=zero_doppler_time,
                dtype=np.float64,
            )
            ds.attrs["units"] = np.bytes_("seconds")

            # Not strictly required by rubbersheet, but useful
            ds = g.create_dataset(
                "slantRangeSpacing",
                data=np.float64(off_grid.range_pixel_spacing),
            )
            ds.attrs["units"] = np.bytes_("meters")

            ds = g.create_dataset(
                "zeroDopplerTimeSpacing",
                data=np.float64(1.0 / off_grid.prf),
            )
            ds.attrs["units"] = np.bytes_("seconds")

            # ----------------------------------------------------------
            # Optional mask
            #
            # rubbersheet reads this only when:
            # subswath_mask_apply_enabled == True
            #
            # 255 matches the native writer's default fill value.
            # ----------------------------------------------------------

            g.create_dataset(
                "mask",
                shape=(off_length, off_width),
                dtype=np.uint32,
                fillvalue=np.uint32(255),
            )

            # ----------------------------------------------------------
            # Datasets that rubbersheet writes into
            # ----------------------------------------------------------

            for pol in pol_list:

                pg = g.require_group(pol)

                ds = pg.create_dataset(
                    "alongTrackOffset",
                    shape=(off_length, off_width),
                    dtype=np.float32,
                    fillvalue=np.nan,
                )
                ds.attrs["units"] = np.bytes_("meters")

                ds = pg.create_dataset(
                    "slantRangeOffset",
                    shape=(off_length, off_width),
                    dtype=np.float32,
                    fillvalue=np.nan,
                )
                ds.attrs["units"] = np.bytes_("meters")

                ds = pg.create_dataset(
                    "correlationSurfacePeak",
                    shape=(off_length, off_width),
                    dtype=np.float32,
                    fillvalue=np.nan,
                )
                ds.attrs["units"] = np.bytes_("1")

            print(
                f"frequency{freq}: "
                f"{off_length} x {off_width}, "
                f"pols={pol_list}"
            )

    print(f"Created minimal rubbersheet HDF5: {out_h5}")

    return str(out_h5)

def set_nested(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def dense_offset_coregistered(
    reference_slc,
    secondary_slc,
    out_dir,
    dense_cfg,
    gpu_id=0,
):
    reference_slc = Path(reference_slc)
    secondary_slc = Path(secondary_slc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU
    device = isce3.cuda.core.Device(gpu_id)
    isce3.cuda.core.set_device(device)

    ampcor = isce3.cuda.matchtemplate.PyCuAmpcor()
    ampcor.deviceID = gpu_id
    ampcor.useMmap = 1

    # Read raster dimensions
    ref_raster = isce3.io.Raster(str(reference_slc))
    sec_raster = isce3.io.Raster(str(secondary_slc))

    if (
        ref_raster.length != sec_raster.length
        or ref_raster.width != sec_raster.width
    ):
        raise ValueError(
            f"Raster dimensions differ: "
            f"ref={ref_raster.length}x{ref_raster.width}, "
            f"sec={sec_raster.length}x{sec_raster.width}"
        )

    # Input files
    ampcor.referenceImageName = str(reference_slc)
    ampcor.referenceImageHeight = ref_raster.length
    ampcor.referenceImageWidth = ref_raster.width

    ampcor.secondaryImageName = str(secondary_slc)
    ampcor.secondaryImageHeight = sec_raster.length
    ampcor.secondaryImageWidth = sec_raster.width

    # Apply normal NISAR dense-offset parameters
    ampcor = set_optional_attributes(
        ampcor,
        dense_cfg,
        ref_raster.length,
        ref_raster.width,
    )

    # Outputs
    ampcor.offsetImageName = str(out_dir / "dense_offsets")
    ampcor.grossOffsetImageName = str(out_dir / "gross_offset")
    ampcor.snrImageName = str(out_dir / "snr")
    ampcor.covImageName = str(out_dir / "covariance")
    ampcor.corrImageName = str(out_dir / "correlation_peak")

    nx = ampcor.numberWindowAcross
    ny = ampcor.numberWindowDown

    create_empty_dataset(
        str(out_dir / "dense_offsets"),
        nx, ny, 2, gdal.GDT_Float32,
    )

    create_empty_dataset(
        str(out_dir / "gross_offset"),
        nx, ny, 2, gdal.GDT_Float32,
    )

    create_empty_dataset(
        str(out_dir / "snr"),
        nx, ny, 1, gdal.GDT_Float32,
    )

    create_empty_dataset(
        str(out_dir / "covariance"),
        nx, ny, 3, gdal.GDT_Float32,
    )

    create_empty_dataset(
        str(out_dir / "correlation_peak"),
        nx, ny, 1, gdal.GDT_Float32,
    )

    print(
        f"Dense offset grid: {ny} × {nx}"
    )

    ampcor.runAmpcor()

    return out_dir / "dense_offsets"