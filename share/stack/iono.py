"""
isce3_pair_iono
===============

Importable pair-level ISCE3 interferogram + split-spectrum ionosphere workflow
for two already-aligned SLC rasters.

Typical use
-----------
from isce3_pair_iono import (
    AlignedPairIonoWorkflow,
    WorkflowConfig,
)

cfg = WorkflowConfig(
    range_looks=11,
    azimuth_looks=11,
    gpu=True,
)

wf = AlignedPairIonoWorkflow(
    ref_slc="/path/reference.slc",
    sec_slc="/path/coregistered_secondary.slc",
    outdir="/path/pair_iono",
    metadata_rslc="/path/reference_rslc.h5",
    ref_rslc="/path/reference_rslc.h5",
    sec_rslc="/path/secondary_rslc.h5",
    config=cfg,
)

products = wf.run()

Notes
-----
The official full NISAR ISCE3 ionosphere workflow splits the ORIGINAL RSLCs
before resampling/coregistering the low/high secondary subbands.

This module instead starts from two already-aligned SLC rasters, so the split
spectrum operation necessarily happens after alignment. From Crossmul onward,
it uses the same ISCE3 module families as the current NISAR workflow:

- isce3.signal.Crossmul
- isce3.splitspectrum.SplitSpectrum
- SNAPHU (default), isce3.unwrap.ICU, or Whirlwind
- LowHighSubbandIonosphereEstimation
- IonosphereFilter
- split-band unwrapping-error estimation/correction
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from osgeo import gdal
from scipy.fft import next_fast_len

import isce3
from isce3.atmosphere.ionosphere_filter import IonosphereFilter
from isce3.atmosphere.split_band_estimation import (
    LowHighSubbandIonosphereEstimation,
)
from isce3.splitspectrum.splitspectrum import BandpassMetaData, SplitSpectrum
from isce3.unwrap.bridge_phase import bridge_unwrapped_phase
from nisar.products.readers import RSLC

gdal.UseExceptions()

C = float(isce3.core.speed_of_light)


@dataclass
class WorkflowConfig:
    """Processing configuration for :class:`AlignedPairIonoWorkflow`."""

    # Crossmul
    range_looks: int = 11
    azimuth_looks: int = 11
    oversample: int = 2
    crossmul_lines_per_block: int = 1024
    gpu: bool = False
    gpu_id: int = 0

    # Optional flattening
    flatten_range_offset: Optional[str] = None
    starting_range_shift: float = 0.0

    # Split spectrum
    split_lines_per_block: int = 2048
    subband_fraction: float = 1.0 / 3.0
    window_function: str = "tukey"
    window_shape: float = 0.25

    # Unwrapping: "snaphu", "icu", or "whirlwind"
    unwrap_algorithm: str = "snaphu"

    # Whirlwind (package: whirlwind-insar; import: whirlwind)
    whirlwind_nlooks: Optional[float] = None
    whirlwind_downsample: Optional[int] = None

    # SNAPHU defaults close to current NISAR ISCE3 defaults
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

    # Bridge disconnected unwrapped regions
    bridge: bool = True
    bridge_radius: int = 500
    bridge_erosion_size: int = 2
    bridge_minimum_samples: int = 14
    bridge_ramp_type: Optional[str] = None
    bridge_ramp_maximum_pixel: int = 1_000_000

    # Ionosphere mask
    iono_mask_types: tuple[str, ...] = ("coherence",)
    iono_coherence_threshold: float = 0.5

    # Ionosphere filtering
    iono_filter: bool = True
    iono_kernel_range: int = 100
    iono_kernel_azimuth: int = 100
    iono_sigma_range: int = 33
    iono_sigma_azimuth: int = 33
    iono_filter_iterations: int = 1
    iono_filling_method: str = "nearest"
    iono_min_cluster_pixels: int = 2

    # Current official-style split-band unwrap-error correction
    iono_unwrap_correction: bool = True

    # File handling
    overwrite: bool = False


@dataclass
class RadarMeta:
    """Radar metadata needed by split-spectrum and Crossmul."""

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
        dr = C / (2.0 * float(range_sample_frequency))
        f0 = float(center_frequency)

        return cls(
            center_frequency=f0,
            range_bandwidth=float(range_bandwidth),
            range_sample_frequency=float(range_sample_frequency),
            range_pixel_spacing=dr,
            starting_range=float(starting_range),
            wavelength=C / f0,
        )

    def slant_range(self, index: int) -> float:
        return self.starting_range + float(index) * self.range_pixel_spacing


@dataclass
class WorkflowProducts:
    """Paths to the principal products generated by the workflow."""

    main_wrapped_interferogram: Path
    main_coherence: Path
    main_unwrapped_phase: Path
    main_connected_components: Path

    low_wrapped_interferogram: Path
    low_coherence: Path
    low_unwrapped_phase: Path
    low_connected_components: Path

    high_wrapped_interferogram: Path
    high_coherence: Path
    high_unwrapped_phase: Path
    high_connected_components: Path

    ionosphere_phase_screen: Path
    ionosphere_uncertainty: Path
    ionosphere_mask: Path
    nondispersive_phase_screen: Path

    main_iono_corrected_unwrapped_phase: Optional[Path] = None


class AlignedPairIonoWorkflow:
    """
    Pair-level ISCE3 workflow for two already-aligned SLC rasters.

    Parameters
    ----------
    ref_slc, sec_slc
        GDAL-readable complex SLC rasters on the same radar grid.
    outdir
        Output directory.
    metadata_rslc
        Optional NISAR RSLC HDF5 used to derive frequency/range metadata.
    radar_meta
        Explicit :class:`RadarMeta`. Use this instead of ``metadata_rslc``.
    ref_rslc, sec_rslc
        Optional original NISAR RSLC HDF5 files. If supplied, their Doppler
        LUTs are passed to Crossmul like the official NISAR workflow.
    frequency
        NISAR frequency, normally ``"A"``.
    config
        :class:`WorkflowConfig`.
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
        config: WorkflowConfig | None = None,
    ):
        self.ref_slc = Path(ref_slc).resolve()
        self.sec_slc = Path(sec_slc).resolve()
        self.outdir = Path(outdir).resolve()

        self.frequency = frequency
        self.config = config or WorkflowConfig()

        self.ref_rslc = Path(ref_rslc).resolve() if ref_rslc else None
        self.sec_rslc = Path(sec_rslc).resolve() if sec_rslc else None

        if radar_meta is not None and metadata_rslc is not None:
            raise ValueError(
                "Provide either radar_meta or metadata_rslc, not both."
            )

        if radar_meta is not None:
            self.meta = radar_meta
        elif metadata_rslc is not None:
            self.meta = RadarMeta.from_nisar_rslc(
                metadata_rslc,
                frequency=frequency,
            )
        else:
            raise ValueError(
                "Provide either metadata_rslc or an explicit RadarMeta."
            )

        self.outdir.mkdir(parents=True, exist_ok=True)

        self.width, self.length = self._validate_aligned()
        self.dopplers = self._load_dopplers()

        (
            self.low_limits,
            self.high_limits,
            self.low_center_frequency,
            self.high_center_frequency,
        ) = self._compute_subbands()

        self._write_workflow_metadata()

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    def run(self) -> WorkflowProducts:
        """Run the complete pair workflow."""

        print("[1/6] Main-band Crossmul")
        main_ifg, main_coh = self.crossmul_main()

        print("[2/6] Main-band unwrap")
        main_unw, main_cc = self.unwrap_main()

        print("[3/6] Split spectrum")
        self.split_aligned_slcs()

        print("[4/6] Low/high Crossmul")
        low_ifg, low_coh, high_ifg, high_coh = self.crossmul_subbands()

        print("[5/6] Low/high unwrap")
        low_unw, low_cc, high_unw, high_cc = self.unwrap_subbands()

        print("[6/6] Ionosphere estimation")
        iono, iono_sigma, iono_mask, nondisp = self.estimate_ionosphere()

        corrected = self.correct_main_unwrapped_phase()

        return WorkflowProducts(
            main_wrapped_interferogram=main_ifg,
            main_coherence=main_coh,
            main_unwrapped_phase=main_unw,
            main_connected_components=main_cc,
            low_wrapped_interferogram=low_ifg,
            low_coherence=low_coh,
            low_unwrapped_phase=low_unw,
            low_connected_components=low_cc,
            high_wrapped_interferogram=high_ifg,
            high_coherence=high_coh,
            high_unwrapped_phase=high_unw,
            high_connected_components=high_cc,
            ionosphere_phase_screen=iono,
            ionosphere_uncertainty=iono_sigma,
            ionosphere_mask=iono_mask,
            nondispersive_phase_screen=nondisp,
            main_iono_corrected_unwrapped_phase=corrected,
        )

    def crossmul_main(self) -> tuple[Path, Path]:
        """Form the main-band wrapped interferogram and coherence."""
        return self._crossmul_pair(
            self.ref_slc,
            self.sec_slc,
            self.outdir / "main",
        )

    def unwrap_main(self) -> tuple[Path, Path]:
        """Unwrap the main-band interferogram."""
        main_dir = self.outdir / "main"
        return self._unwrap_pair(
            main_dir / "wrappedInterferogram.int",
            main_dir / "coherenceMagnitude.cor",
            main_dir,
        )

    def split_aligned_slcs(self) -> dict[str, Path]:
        """Split the two aligned SLCs into low/high range subbands."""

        split_dir = self.outdir / "split_spectrum"
        split_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "ref_low": split_dir / "ref_low.slc",
            "ref_high": split_dir / "ref_high.slc",
            "sec_low": split_dir / "sec_low.slc",
            "sec_high": split_dir / "sec_high.slc",
        }

        self._split_slc(
            self.ref_slc,
            paths["ref_low"],
            paths["ref_high"],
        )
        self._split_slc(
            self.sec_slc,
            paths["sec_low"],
            paths["sec_high"],
        )

        return paths

    def crossmul_subbands(
        self,
    ) -> tuple[Path, Path, Path, Path]:
        """Form low/high subband interferograms and coherences."""

        split_dir = self.outdir / "split_spectrum"

        low_ifg, low_coh = self._crossmul_pair(
            split_dir / "ref_low.slc",
            split_dir / "sec_low.slc",
            self.outdir / "low",
        )

        high_ifg, high_coh = self._crossmul_pair(
            split_dir / "ref_high.slc",
            split_dir / "sec_high.slc",
            self.outdir / "high",
        )

        return low_ifg, low_coh, high_ifg, high_coh

    def unwrap_subbands(
        self,
    ) -> tuple[Path, Path, Path, Path]:
        """Unwrap low/high subband interferograms."""

        low_dir = self.outdir / "low"
        high_dir = self.outdir / "high"

        low_unw, low_cc = self._unwrap_pair(
            low_dir / "wrappedInterferogram.int",
            low_dir / "coherenceMagnitude.cor",
            low_dir,
        )

        high_unw, high_cc = self._unwrap_pair(
            high_dir / "wrappedInterferogram.int",
            high_dir / "coherenceMagnitude.cor",
            high_dir,
        )

        return low_unw, low_cc, high_unw, high_cc

    def estimate_ionosphere(
        self,
    ) -> tuple[Path, Path, Path, Path]:
        """
        Estimate, mask, optionally unwrap-correct, and filter the ionosphere.
        """

        cfg = self.config
        low_dir = self.outdir / "low"
        high_dir = self.outdir / "high"
        iono_dir = self.outdir / "ionosphere"
        iono_dir.mkdir(parents=True, exist_ok=True)

        phi_low = self._read_array(
            low_dir / "unwrappedPhase.unw",
            np.float64,
        )
        phi_high = self._read_array(
            high_dir / "unwrappedPhase.unw",
            np.float64,
        )

        low_coh = self._read_array(
            low_dir / "coherenceMagnitude.cor",
            np.float32,
        )
        high_coh = self._read_array(
            high_dir / "coherenceMagnitude.cor",
            np.float32,
        )

        low_cc = self._read_array(
            low_dir / "connectedComponents.conncomp"
        )
        high_cc = self._read_array(
            high_dir / "connectedComponents.conncomp"
        )

        if cfg.bridge:
            phi_low = self._bridge_phase(phi_low)
            phi_high = self._bridge_phase(phi_high)

        estimator = LowHighSubbandIonosphereEstimation(
            main_center_freq=self.meta.center_frequency,
            low_center_freq=self.low_center_frequency,
            high_center_freq=self.high_center_frequency,
        )

        dispersive, nondispersive = estimator.compute_disp_nondisp(
            phi_sub_low=phi_low,
            phi_sub_high=phi_high,
            no_data=0,
        )

        nlooks = cfg.range_looks * cfg.azimuth_looks

        iono_sigma, nondisp_sigma = estimator.estimate_iono_std(
            low_band_coh=low_coh,
            high_band_coh=high_coh,
            number_looks=nlooks,
            resample_flag=False,
        )

        mask = self._make_iono_mask(
            estimator,
            phi_low,
            phi_high,
            low_coh,
            high_coh,
            low_cc,
            high_cc,
        )

        raw_disp = iono_dir / "dispersive.raw"
        raw_non = iono_dir / "nonDispersive.raw"
        raw_sig = iono_dir / "dispersive.sig"
        raw_non_sig = iono_dir / "nonDispersive.sig"
        mask_path = iono_dir / "ionoValidMask.mask"

        self._write_array(
            raw_disp,
            dispersive.astype(np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            raw_non,
            nondispersive.astype(np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            raw_sig,
            iono_sigma.astype(np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            raw_non_sig,
            nondisp_sigma.astype(np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            mask_path,
            mask.astype(np.uint8),
            gdal.GDT_Byte,
        )

        final_disp = iono_dir / "ionospherePhaseScreen"
        final_sig = iono_dir / "ionospherePhaseScreenUncertainty"
        final_non = iono_dir / "nonDispersivePhaseScreen"
        final_non_sig = iono_dir / "nonDispersivePhaseScreenUncertainty"

        if not cfg.iono_filter:
            self._write_array(
                final_disp,
                np.where(mask, dispersive, 0).astype(np.float32),
                gdal.GDT_Float32,
            )
            self._write_array(
                final_sig,
                np.where(mask, iono_sigma, 0).astype(np.float32),
                gdal.GDT_Float32,
            )
            self._write_array(
                final_non,
                np.where(mask, nondispersive, 0).astype(np.float32),
                gdal.GDT_Float32,
            )
            self._write_array(
                final_non_sig,
                np.where(mask, nondisp_sigma, 0).astype(np.float32),
                gdal.GDT_Float32,
            )
            return final_disp, final_sig, mask_path, final_non

        if cfg.iono_unwrap_correction:
            prelim_disp = iono_dir / "dispersive.prelim_filt"
            prelim_disp_sig = iono_dir / "dispersive.prelim_filt.sig"
            prelim_non = iono_dir / "nonDispersive.prelim_filt"
            prelim_non_sig = iono_dir / "nonDispersive.prelim_filt.sig"

            self._run_iono_filter(
                raw_disp,
                raw_sig,
                mask_path,
                prelim_disp,
                prelim_disp_sig,
                iono_dir / "filter_prelim_disp",
            )
            self._run_iono_filter(
                raw_non,
                raw_non_sig,
                mask_path,
                prelim_non,
                prelim_non_sig,
                iono_dir / "filter_prelim_non",
            )

            filt_disp = self._read_array(prelim_disp, np.float64)
            filt_non = self._read_array(prelim_non, np.float64)

            common_coef, diff_coef = estimator.compute_unwrapp_error(
                disp_array=filt_disp,
                nondisp_array=filt_non,
                low_sub_runw=phi_low,
                high_sub_runw=phi_high,
            )

            self._write_array(
                iono_dir / "commonUnwrapErrorCoefficient",
                common_coef.astype(np.int32),
                gdal.GDT_Int32,
            )
            self._write_array(
                iono_dir / "differentialUnwrapErrorCoefficient",
                diff_coef.astype(np.int32),
                gdal.GDT_Int32,
            )

            corrected_disp, corrected_non = estimator.compute_disp_nondisp(
                phi_sub_low=phi_low,
                phi_sub_high=phi_high,
                comm_unwcor_coef=common_coef,
                diff_unwcor_coef=diff_coef,
                no_data=0,
            )

            corr_disp = iono_dir / "dispersive.unwrap_corrected"
            corr_non = iono_dir / "nonDispersive.unwrap_corrected"

            self._write_array(
                corr_disp,
                corrected_disp.astype(np.float32),
                gdal.GDT_Float32,
            )
            self._write_array(
                corr_non,
                corrected_non.astype(np.float32),
                gdal.GDT_Float32,
            )

            self._run_iono_filter(
                corr_disp,
                raw_sig,
                mask_path,
                final_disp,
                final_sig,
                iono_dir / "filter_final_disp",
            )
            self._run_iono_filter(
                corr_non,
                raw_non_sig,
                mask_path,
                final_non,
                final_non_sig,
                iono_dir / "filter_final_non",
            )

        else:
            self._run_iono_filter(
                raw_disp,
                raw_sig,
                mask_path,
                final_disp,
                final_sig,
                iono_dir / "filter_final_disp",
            )
            self._run_iono_filter(
                raw_non,
                raw_non_sig,
                mask_path,
                final_non,
                final_non_sig,
                iono_dir / "filter_final_non",
            )

        return final_disp, final_sig, mask_path, final_non

    def correct_main_unwrapped_phase(self) -> Optional[Path]:
        """
        Subtract the estimated ionosphere screen from the main unwrapped phase.
        """

        main_path = self.outdir / "main" / "unwrappedPhase.unw"
        iono_path = self.outdir / "ionosphere" / "ionospherePhaseScreen"

        if not main_path.exists() or not iono_path.exists():
            return None

        main = self._read_array(main_path, np.float32)
        iono = self._read_array(iono_path, np.float32)

        if main.shape != iono.shape:
            return None

        valid = np.isfinite(main) & np.isfinite(iono)

        corrected = np.zeros_like(main, dtype=np.float32)
        corrected[valid] = main[valid] - iono[valid]

        out = (
            self.outdir
            / "main"
            / "unwrappedPhase_ionoCorrected.unw"
        )

        self._write_array(
            out,
            corrected,
            gdal.GDT_Float32,
        )

        return out

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _validate_aligned(self) -> tuple[int, int]:
        ref = self._open_gdal(self.ref_slc)
        sec = self._open_gdal(self.sec_slc)

        ref_shape = (ref.RasterXSize, ref.RasterYSize)
        sec_shape = (sec.RasterXSize, sec.RasterYSize)

        if ref_shape != sec_shape:
            raise ValueError(
                f"SLC dimensions differ: REF={ref_shape}, SEC={sec_shape}"
            )

        ref = None
        sec = None
        return ref_shape

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

    def _compute_subbands(self):
        cfg = self.config
        f0 = self.meta.center_frequency
        bw = self.meta.range_bandwidth
        sub_bw = bw * cfg.subband_fraction

        if not (0 < sub_bw <= bw / 2):
            raise ValueError(
                "subband_fraction produces an invalid subband bandwidth"
            )

        f_min = f0 - bw / 2
        f_max = f0 + bw / 2

        low_limits = (f_min, f_min + sub_bw)
        high_limits = (f_max - sub_bw, f_max)

        f_low = 0.5 * sum(low_limits)
        f_high = 0.5 * sum(high_limits)

        return low_limits, high_limits, f_low, f_high

    def _write_workflow_metadata(self):
        metadata = {
            "ref_slc": str(self.ref_slc),
            "sec_slc": str(self.sec_slc),
            "width": self.width,
            "length": self.length,
            "frequency": self.frequency,
            "radar_meta": asdict(self.meta),
            "config": asdict(self.config),
            "low_band_limits_hz": self.low_limits,
            "high_band_limits_hz": self.high_limits,
            "low_center_frequency_hz": self.low_center_frequency,
            "high_center_frequency_hz": self.high_center_frequency,
            "note": (
                "Pair-level aligned-SLC workflow. Official full NISAR ISCE3 "
                "splits original RSLCs before subband resampling/coregistration."
            ),
        }

        (self.outdir / "workflow_metadata.json").write_text(
            json.dumps(metadata, indent=2)
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
        cm.lines_per_block = cfg.crossmul_lines_per_block

        cm.range_pixel_spacing = self.meta.range_pixel_spacing
        cm.wavelength = self.meta.wavelength

        if self.dopplers is not None:
            cm.set_dopplers(*self.dopplers)

        if cfg.flatten_range_offset:
            cm.ref_sec_offset_starting_range_shift = (
                cfg.starting_range_shift
            )

        return cm

    def _crossmul_pair(
        self,
        ref_path: Path,
        sec_path: Path,
        outdir: Path,
    ) -> tuple[Path, Path]:
        cfg = self.config
        outdir.mkdir(parents=True, exist_ok=True)

        ifg_path = outdir / "wrappedInterferogram.int"
        coh_path = outdir / "coherenceMagnitude.cor"

        self._prepare_output(ifg_path)
        self._prepare_output(coh_path)

        ref = isce3.io.Raster(str(ref_path))
        sec = isce3.io.Raster(str(sec_path))

        if ref.width != sec.width or ref.length != sec.length:
            raise ValueError(
                f"Grid mismatch: {ref_path} vs {sec_path}"
            )

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

    def _split_slc(
        self,
        src_path: Path,
        low_path: Path,
        high_path: Path,
    ):
        cfg = self.config

        self._prepare_output(low_path)
        self._prepare_output(high_path)

        src = self._open_gdal(src_path)
        width = src.RasterXSize
        length = src.RasterYSize

        driver = gdal.GetDriverByName("ENVI")

        low_ds = driver.Create(
            str(low_path),
            width,
            length,
            1,
            gdal.GDT_CFloat32,
        )
        high_ds = driver.Create(
            str(high_path),
            width,
            length,
            1,
            gdal.GDT_CFloat32,
        )

        splitter = SplitSpectrum(
            rg_sample_freq=self.meta.range_sample_frequency,
            rg_bandwidth=self.meta.range_bandwidth,
            center_frequency=self.meta.center_frequency,
            slant_range=self.meta.slant_range,
            freq=self.frequency,
        )

        fft_size = next_fast_len(width)

        src_band = src.GetRasterBand(1)
        low_band = low_ds.GetRasterBand(1)
        high_band = high_ds.GetRasterBand(1)

        for y0 in range(0, length, cfg.split_lines_per_block):
            nlines = min(
                cfg.split_lines_per_block,
                length - y0,
            )

            block = src_band.ReadAsArray(
                0,
                y0,
                width,
                nlines,
            ).astype(
                np.complex64,
                copy=False,
            )

            low, _ = splitter.bandpass_shift_spectrum(
                slc_raster=block,
                low_frequency=self.low_limits[0],
                high_frequency=self.low_limits[1],
                new_center_frequency=self.low_center_frequency,
                window_function=cfg.window_function,
                window_shape=cfg.window_shape,
                fft_size=fft_size,
                resampling=False,
            )

            high, _ = splitter.bandpass_shift_spectrum(
                slc_raster=block,
                low_frequency=self.high_limits[0],
                high_frequency=self.high_limits[1],
                new_center_frequency=self.high_center_frequency,
                window_function=cfg.window_function,
                window_shape=cfg.window_shape,
                fft_size=fft_size,
                resampling=False,
            )

            low_band.WriteArray(
                np.asarray(low, dtype=np.complex64),
                0,
                y0,
            )
            high_band.WriteArray(
                np.asarray(high, dtype=np.complex64),
                0,
                y0,
            )

        low_ds.FlushCache()
        high_ds.FlushCache()

        low_ds = None
        high_ds = None
        src = None

    def _unwrap_pair(
        self,
        ifg_path: Path,
        coh_path: Path,
        outdir: Path,
    ) -> tuple[Path, Path]:
        algo = self.config.unwrap_algorithm.lower()

        if algo == "snaphu":
            return self._unwrap_snaphu(ifg_path, coh_path, outdir)

        if algo == "icu":
            return self._unwrap_icu(ifg_path, coh_path, outdir)

        if algo == "whirlwind":
            return self._unwrap_whirlwind(ifg_path, coh_path, outdir)

        raise ValueError(
            f"Unsupported unwrap_algorithm={self.config.unwrap_algorithm!r}. "
            "Choose 'snaphu', 'icu', or 'whirlwind'."
        )

    def _unwrap_snaphu(
        self,
        ifg_path: Path,
        coh_path: Path,
        outdir: Path,
    ) -> tuple[Path, Path]:
        import snaphu

        cfg = self.config
        outdir.mkdir(parents=True, exist_ok=True)

        unw_path = outdir / "unwrappedPhase.unw"
        cc_path = outdir / "connectedComponents.conncomp"

        self._prepare_output(unw_path)
        self._prepare_output(cc_path)

        igram = self._read_array(
            ifg_path,
            np.complex64,
        )
        coh = self._read_array(
            coh_path,
            np.float32,
        )

        nlooks = (
            cfg.snaphu_nlooks
            if cfg.snaphu_nlooks is not None
            else float(cfg.range_looks * cfg.azimuth_looks)
        )

        scratch = outdir / "snaphu_scratch"

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

        self._write_array(
            unw_path,
            np.asarray(unw, dtype=np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            cc_path,
            np.asarray(conncomp, dtype=np.uint32),
            gdal.GDT_UInt32,
        )

        return unw_path, cc_path

    def _unwrap_whirlwind(
        self,
        ifg_path: Path,
        coh_path: Path,
        outdir: Path,
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
        outdir.mkdir(parents=True, exist_ok=True)

        unw_path = outdir / "unwrappedPhase.unw"
        cc_path = outdir / "connectedComponents.conncomp"

        self._prepare_output(unw_path)
        self._prepare_output(cc_path)

        igram = self._read_array(ifg_path, np.complex64)
        coh = self._read_array(coh_path, np.float32)

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

        self._write_array(
            unw_path,
            np.asarray(unw, dtype=np.float32),
            gdal.GDT_Float32,
        )
        self._write_array(
            cc_path,
            np.asarray(conncomp, dtype=np.uint32),
            gdal.GDT_UInt32,
        )

        return unw_path, cc_path

    def _unwrap_icu(
        self,
        ifg_path: Path,
        coh_path: Path,
        outdir: Path,
    ) -> tuple[Path, Path]:
        unw_path = outdir / "unwrappedPhase.unw"
        cc_path = outdir / "connectedComponents.conncomp"

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
        icu.unwrap(
            unw,
            cc,
            igram,
            coh,
        )

        del unw
        del cc
        del igram
        del coh

        return unw_path, cc_path

    def _bridge_phase(self, phase):
        cfg = self.config

        return bridge_unwrapped_phase(
            phase,
            radius=cfg.bridge_radius,
            min_num_pixel=cfg.bridge_minimum_samples,
            erosion_size=cfg.bridge_erosion_size,
            ramp_type=cfg.bridge_ramp_type,
            deramp_max_num_sample=cfg.bridge_ramp_maximum_pixel,
        )

    def _make_iono_mask(
        self,
        estimator,
        phi_low,
        phi_high,
        low_coh,
        high_coh,
        low_cc,
        high_cc,
    ):
        cfg = self.config
        mask = np.ones(phi_low.shape, dtype=bool)

        if "coherence" in cfg.iono_mask_types:
            mask &= estimator.get_coherence_mask_array(
                low_band_array=low_coh,
                high_band_array=high_coh,
                threshold=cfg.iono_coherence_threshold,
            )

        if "connected_components" in cfg.iono_mask_types:
            mask &= estimator.get_conn_component_mask_array(
                low_band_array=low_cc,
                high_band_array=high_cc,
            )

        mask &= estimator.get_valid_area(
            low_band_array=phi_low,
            high_band_array=phi_high,
            invalid_value=0,
        )

        mask &= estimator.get_valid_area(
            low_band_array=low_coh,
            high_band_array=high_coh,
            invalid_value=0,
        )

        mask &= np.isfinite(phi_low)
        mask &= np.isfinite(phi_high)
        mask &= np.isfinite(low_coh)
        mask &= np.isfinite(high_coh)

        return mask

    def _run_iono_filter(
        self,
        data_path: Path,
        sigma_path: Path,
        mask_path: Path,
        out_path: Path,
        out_sigma_path: Path,
        filter_dir: Path,
    ):
        cfg = self.config
        filter_dir.mkdir(parents=True, exist_ok=True)

        filt = IonosphereFilter(
            x_kernel=cfg.iono_kernel_range,
            y_kernel=cfg.iono_kernel_azimuth,
            sig_x=cfg.iono_sigma_range,
            sig_y=cfg.iono_sigma_azimuth,
            iteration=cfg.iono_filter_iterations,
            filling_method=cfg.iono_filling_method,
            guide_filter_method="median_gaussian",
            guide_median_size=3,
            outlier_threshold=3.5,
            outlier_min_scale=0.0,
            mad_scale_factor=1.4826,
            outputdir=str(filter_dir),
        )

        self._remove_envi(out_path)
        self._remove_envi(out_sigma_path)

        filt.low_pass_filter(
            input_data=str(data_path),
            input_std_dev=str(sigma_path),
            mask_path=str(mask_path),
            filtered_output=str(out_path),
            filtered_std_dev=str(out_sigma_path),
            lines_per_block=1000,
            min_cluster_pixels=cfg.iono_min_cluster_pixels,
        )

    @staticmethod
    def _open_gdal(path: str | Path):
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)

        if ds is None:
            raise RuntimeError(
                f"GDAL cannot open {path}"
            )

        return ds

    @classmethod
    def _read_array(cls, path: str | Path, dtype=None):
        ds = cls._open_gdal(path)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)

        return arr

    def _write_array(
        self,
        path: str | Path,
        array,
        gdal_type,
    ):
        path = Path(path)
        self._prepare_output(path)

        path.parent.mkdir(parents=True, exist_ok=True)

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
                    f"{path} exists. Set WorkflowConfig(overwrite=True) "
                    "to replace existing outputs."
                )

            self._remove_envi(path)

    @staticmethod
    def _remove_envi(path: str | Path):
        path = Path(path)

        for target in (
            path,
            Path(str(path) + ".hdr"),
            Path(str(path) + ".aux.xml"),
        ):
            if target.exists():
                target.unlink()


__all__ = [
    "AlignedPairIonoWorkflow",
    "RadarMeta",
    "WorkflowConfig",
    "WorkflowProducts",
]
