"""
isce3_pair_igram
================

Pair-level interferogram workflow for two already-aligned SLC rasters.

Provides:
- ISCE3 Crossmul
- wrapped interferogram
- coherence
- SNAPHU, ICU, or Whirlwind unwrapping
- connected components

Example
-------
from isce3_pair_igram import (
    PairIgramWorkflow,
    IgramConfig,
    RadarMeta,
)

cfg = IgramConfig(
    range_looks=11,
    azimuth_looks=11,
    gpu=True,
    overwrite=True,
)

wf = PairIgramWorkflow(
    ref_slc="/path/reference.slc",
    sec_slc="/path/coregistered_secondary.slc",
    outdir="/path/pair_igram",
    metadata_rslc="/path/reference_rslc.h5",
    ref_rslc="/path/reference_rslc.h5",
    sec_rslc="/path/secondary_rslc.h5",
    config=cfg,
)

products = wf.run()
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from osgeo import gdal

import isce3
from isce3.splitspectrum.splitspectrum import BandpassMetaData
from nisar.products.readers import RSLC

gdal.UseExceptions()

C = float(isce3.core.speed_of_light)


@dataclass
class RadarMeta:
    center_frequency: float
    range_bandwidth: float
    range_sample_frequency: float
    range_pixel_spacing: float
    starting_range: float
    wavelength: float

    @classmethod
    def from_nisar_rslc(cls, rslc_path: str | Path, frequency: str = "A"):
        slc = RSLC(hdf5file=str(rslc_path))
        meta = BandpassMetaData.load_from_slc(
            slc_product=slc,
            freq=frequency,
        )
        grid = slc.getRadarGrid(frequency)

        return cls(
            center_frequency=float(meta.center_freq),
            range_bandwidth=float(meta.rg_bandwidth),
            range_sample_frequency=float(meta.rg_sample_freq),
            range_pixel_spacing=float(meta.rg_pxl_spacing),
            starting_range=float(grid.starting_range),
            wavelength=float(meta.wavelength),
        )

    @classmethod
    def from_values(
        cls,
        *,
        center_frequency: float,
        range_bandwidth: float,
        range_sample_frequency: float,
        starting_range: float = 0.0,
    ):
        fs = float(range_sample_frequency)
        dr = C / (2.0 * fs)
        f0 = float(center_frequency)

        return cls(
            center_frequency=f0,
            range_bandwidth=float(range_bandwidth),
            range_sample_frequency=fs,
            range_pixel_spacing=dr,
            starting_range=float(starting_range),
            wavelength=C / f0,
        )

    def slant_range(self, index: int) -> float:
        return self.starting_range + float(index) * self.range_pixel_spacing


@dataclass
class IgramConfig:
    # Crossmul
    range_looks: int = 11
    azimuth_looks: int = 11
    oversample: int = 2
    lines_per_block: int = 1024

    gpu: bool = False
    gpu_id: int = 0

    # Optional flattening
    flatten_range_offset: Optional[str] = None
    starting_range_shift: float = 0.0

    # Unwrapping: "snaphu", "icu", or "whirlwind"
    unwrap_algorithm: str = "snaphu"

    # Whirlwind (package: whirlwind-insar; import: whirlwind)
    whirlwind_nlooks: Optional[float] = None
    whirlwind_downsample: Optional[int] = None

    snaphu_nlooks: Optional[float] = None
    snaphu_cost: str = "smooth"
    snaphu_init: str = "mcf"
    snaphu_min_conncomp_frac: float = 0.01
    snaphu_phase_grad_window: tuple[int, int] = (7, 7)
    snaphu_ntiles: tuple[int, int] = (1, 1)
    snaphu_tile_overlap: tuple[int, int] = (0, 0)
    snaphu_nproc: int = 1
    snaphu_tile_cost_thresh: float = 500.0
    snaphu_min_region_size: int = 300
    snaphu_single_tile_reoptimize: bool = True

    overwrite: bool = False


@dataclass
class IgramProducts:
    wrapped_interferogram: Path
    coherence: Path
    unwrapped_phase: Path
    connected_components: Path


class PairIgramWorkflow:
    """
    Crossmul + coherence + unwrap for an already-aligned SLC pair.
    """

    def __init__(
        self,
        *,
        ref_slc: str | Path,
        sec_slc: str | Path,
        outdir: str | Path,
        metadata_rslc: str | Path | None = None,
        radar_meta: RadarMeta | None = None,
        ref_rslc: str | Path | None = None,
        sec_rslc: str | Path | None = None,
        frequency: str = "A",
        config: IgramConfig | None = None,
    ):
        self.ref_slc = Path(ref_slc).resolve()
        self.sec_slc = Path(sec_slc).resolve()
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.frequency = frequency
        self.config = config or IgramConfig()

        self.ref_rslc = Path(ref_rslc).resolve() if ref_rslc else None
        self.sec_rslc = Path(sec_rslc).resolve() if sec_rslc else None

        if metadata_rslc is not None and radar_meta is not None:
            raise ValueError("Provide either metadata_rslc or radar_meta, not both.")

        if radar_meta is not None:
            self.meta = radar_meta
        elif metadata_rslc is not None:
            self.meta = RadarMeta.from_nisar_rslc(
                metadata_rslc,
                frequency=frequency,
            )
        else:
            raise ValueError("Provide metadata_rslc or radar_meta.")

        self.width, self.length = self._validate_aligned()
        self.dopplers = self._load_dopplers()

    def run(self) -> IgramProducts:
        ifg, coh = self.crossmul()
        unw, cc = self.unwrap()

        return IgramProducts(
            wrapped_interferogram=ifg,
            coherence=coh,
            unwrapped_phase=unw,
            connected_components=cc,
        )

    def crossmul(self) -> tuple[Path, Path]:
        cfg = self.config

        ifg_path = self.outdir / "wrappedInterferogram.int"
        coh_path = self.outdir / "coherenceMagnitude.cor"

        self._prepare_output(ifg_path)
        self._prepare_output(coh_path)

        ref = isce3.io.Raster(str(self.ref_slc))
        sec = isce3.io.Raster(str(self.sec_slc))

        if ref.width != sec.width or ref.length != sec.length:
            raise ValueError("Reference and secondary SLC grids differ.")

        out_width = ref.width // cfg.range_looks
        out_length = ref.length // cfg.azimuth_looks

        ifg = isce3.io.Raster(
            str(ifg_path),
            out_width,
            out_length,
            1,
            gdal.GDT_CFloat32,
            "ENVI",
        )
        coh = isce3.io.Raster(
            str(coh_path),
            out_width,
            out_length,
            1,
            gdal.GDT_Float32,
            "ENVI",
        )

        cm = self._new_crossmul()

        flatten = (
            isce3.io.Raster(str(cfg.flatten_range_offset))
            if cfg.flatten_range_offset
            else None
        )

        cm.crossmul(
            ref,
            sec,
            ifg,
            coh,
            flatten,
        )

        del ifg
        del coh
        del ref
        del sec

        if flatten is not None:
            del flatten

        return ifg_path, coh_path

    def unwrap(self) -> tuple[Path, Path]:
        algo = self.config.unwrap_algorithm.lower()

        ifg = self.outdir / "wrappedInterferogram.int"
        coh = self.outdir / "coherenceMagnitude.cor"

        if algo == "snaphu":
            return self._unwrap_snaphu(ifg, coh)

        if algo == "icu":
            return self._unwrap_icu(ifg, coh)

        if algo == "whirlwind":
            return self._unwrap_whirlwind(ifg, coh)

        raise ValueError(
            f"Unsupported unwrap algorithm: {algo!r}. "
            "Choose 'snaphu', 'icu', or 'whirlwind'."
        )

    def _new_crossmul(self):
        cfg = self.config

        if cfg.gpu:
            device = isce3.cuda.core.Device(cfg.gpu_id)
            isce3.cuda.core.set_device(device)
            cm = isce3.cuda.signal.Crossmul()
        else:
            cm = isce3.signal.Crossmul()

        cm.range_looks = cfg.range_looks
        cm.az_looks = cfg.azimuth_looks
        cm.oversample_factor = cfg.oversample
        cm.lines_per_block = cfg.lines_per_block

        cm.range_pixel_spacing = self.meta.range_pixel_spacing
        cm.wavelength = self.meta.wavelength

        if self.dopplers is not None:
            cm.set_dopplers(*self.dopplers)

        if cfg.flatten_range_offset:
            cm.ref_sec_offset_starting_range_shift = cfg.starting_range_shift

        return cm

    def _load_dopplers(self):
        if self.ref_rslc is None or self.sec_rslc is None:
            return None

        ref = RSLC(hdf5file=str(self.ref_rslc))
        sec = RSLC(hdf5file=str(self.sec_rslc))

        ref_dopp = isce3.core.avg_lut2d_to_lut1d(
            ref.getDopplerCentroid(frequency=self.frequency)
        )
        sec_dopp = isce3.core.avg_lut2d_to_lut1d(
            sec.getDopplerCentroid(frequency=self.frequency)
        )

        return ref_dopp, sec_dopp

    def _unwrap_snaphu(
        self,
        ifg_path: Path,
        coh_path: Path,
    ) -> tuple[Path, Path]:
        import snaphu

        cfg = self.config

        unw_path = self.outdir / "unwrappedPhase.unw"
        cc_path = self.outdir / "connectedComponents.conncomp"

        self._prepare_output(unw_path)
        self._prepare_output(cc_path)

        igram = self.read_array(ifg_path, np.complex64)
        coh = self.read_array(coh_path, np.float32)

        nlooks = (
            cfg.snaphu_nlooks
            if cfg.snaphu_nlooks is not None
            else float(cfg.range_looks * cfg.azimuth_looks)
        )

        scratch = self.outdir / "snaphu_scratch"

        if scratch.exists() and cfg.overwrite:
            shutil.rmtree(scratch)

        scratch.mkdir(parents=True, exist_ok=True)

        unw, conncomp = snaphu.unwrap(
            igram,
            coh,
            nlooks=nlooks,
            cost=cfg.snaphu_cost,
            init=cfg.snaphu_init,
            min_conncomp_frac=cfg.snaphu_min_conncomp_frac,
            phase_grad_window=cfg.snaphu_phase_grad_window,
            ntiles=cfg.snaphu_ntiles,
            tile_overlap=cfg.snaphu_tile_overlap,
            nproc=cfg.snaphu_nproc,
            tile_cost_thresh=cfg.snaphu_tile_cost_thresh,
            min_region_size=cfg.snaphu_min_region_size,
            single_tile_reoptimize=cfg.snaphu_single_tile_reoptimize,
            scratchdir=scratch,
        )

        self.write_array(
            unw_path,
            np.asarray(unw, dtype=np.float32),
            gdal.GDT_Float32,
        )
        self.write_array(
            cc_path,
            np.asarray(conncomp, dtype=np.uint32),
            gdal.GDT_UInt32,
        )

        return unw_path, cc_path

    def _unwrap_whirlwind(
        self,
        ifg_path: Path,
        coh_path: Path,
    ) -> tuple[Path, Path]:
        """Unwrap with the ``whirlwind-insar`` Python package."""
        try:
            import whirlwind as ww
        except ImportError as exc:
            raise ImportError(
                "Whirlwind unwrapping requires the 'whirlwind-insar' package. "
                "Install it with `pip install whirlwind-insar` or "
                "`conda install -c conda-forge whirlwind-insar`."
            ) from exc

        cfg = self.config
        unw_path = self.outdir / "unwrappedPhase.unw"
        cc_path = self.outdir / "connectedComponents.conncomp"

        self._prepare_output(unw_path)
        self._prepare_output(cc_path)

        igram = self.read_array(ifg_path, np.complex64)
        coh = self.read_array(coh_path, np.float32)

        nlooks = (
            cfg.whirlwind_nlooks
            if cfg.whirlwind_nlooks is not None
            else float(cfg.range_looks * cfg.azimuth_looks)
        )

        kwargs = {"nlooks": float(nlooks)}
        if cfg.whirlwind_downsample is not None:
            if cfg.whirlwind_downsample < 1:
                raise ValueError("whirlwind_downsample must be >= 1")
            kwargs["downsample"] = int(cfg.whirlwind_downsample)

        unw, conncomp = ww.unwrap(igram, coh, **kwargs)

        self.write_array(
            unw_path,
            np.asarray(unw, dtype=np.float32),
            gdal.GDT_Float32,
        )
        self.write_array(
            cc_path,
            np.asarray(conncomp, dtype=np.uint32),
            gdal.GDT_UInt32,
        )

        return unw_path, cc_path

    def _unwrap_icu(
        self,
        ifg_path: Path,
        coh_path: Path,
    ) -> tuple[Path, Path]:
        unw_path = self.outdir / "unwrappedPhase.unw"
        cc_path = self.outdir / "connectedComponents.conncomp"

        self._prepare_output(unw_path)
        self._prepare_output(cc_path)

        igram = isce3.io.Raster(str(ifg_path))
        coh = isce3.io.Raster(str(coh_path))

        unw = isce3.io.Raster(
            str(unw_path),
            igram.width,
            igram.length,
            1,
            gdal.GDT_Float32,
            "ENVI",
        )
        cc = isce3.io.Raster(
            str(cc_path),
            igram.width,
            igram.length,
            1,
            gdal.GDT_Byte,
            "ENVI",
        )

        icu = isce3.unwrap.ICU()
        icu.unwrap(unw, cc, igram, coh)

        del unw
        del cc
        del igram
        del coh

        return unw_path, cc_path

    def _validate_aligned(self):
        ref = self._open_gdal(self.ref_slc)
        sec = self._open_gdal(self.sec_slc)

        a = (ref.RasterXSize, ref.RasterYSize)
        b = (sec.RasterXSize, sec.RasterYSize)

        if a != b:
            raise ValueError(f"SLC dimensions differ: REF={a}, SEC={b}")

        ref = None
        sec = None

        return a

    @staticmethod
    def _open_gdal(path):
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL cannot open {path}")
        return ds

    @classmethod
    def read_array(cls, path, dtype=None):
        ds = cls._open_gdal(path)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)

        return arr

    def write_array(self, path, array, gdal_type):
        path = Path(path)
        self._prepare_output(path)

        arr = np.asarray(array)
        length, width = arr.shape

        ds = gdal.GetDriverByName("ENVI").Create(
            str(path),
            width,
            length,
            1,
            gdal_type,
        )
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None

    def _prepare_output(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"{path} exists. Set overwrite=True."
                )
            self.remove_envi(path)

    @staticmethod
    def remove_envi(path):
        path = Path(path)
        for target in (
            path,
            Path(str(path) + ".hdr"),
            Path(str(path) + ".aux.xml"),
        ):
            if target.exists():
                target.unlink()


__all__ = [
    "IgramConfig",
    "IgramProducts",
    "PairIgramWorkflow",
    "RadarMeta",
]
