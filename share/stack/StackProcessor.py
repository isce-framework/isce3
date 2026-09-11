from itertools import combinations
import os
from pathlib import Path
import pickle
import shutil

from nisar.workflows.rubbersheet import run_rubbersheet_with_interpolation
from nisar.workflows.helpers import copy_raster

from coarse_alignment import align_secondary, extract_ref_topo
from invert_offsets import invert_offsets_isce3
from offset_util import (
    create_minimal_rifg_for_rubbersheet,
    dense_offset_coregistered,
)
from ScanList import ScanList
from stack_utils import (
    baseline,
    load_config,
    relative_symlink_contents,
    resample,
    set_nested,
)


pol_freq = {"A": ["HH"]}
runcfg = load_config()
cfg = runcfg.cfg["runconfig"]["groups"]
dense_cfg = cfg["processing"]["dense_offsets"]
rdr2geo_cfg = cfg["processing"]["rdr2geo"]
rubbersheet_cfg = cfg["processing"]["rubbersheet"]


class StackProcessor(ScanList):
    def __init__(
        self,
        slc_folder,
        out_folder,
        dem_file,
        max_baseline_distance=500,
        max_date_apart=30,
        freq="A",
        pol="HH",
        overwrite=False,
        debug=False,
    ):
        self.slc_folder = Path(slc_folder)
        self.out_folder = Path(out_folder)
        self.dem_file = Path(dem_file)
        self.overwrite = overwrite
        self.freq = freq
        self.pol = pol
        self.debug = debug

        self.config_save = self.get_processor_config_path()

        h5_files = [str(i) for i in self.slc_folder.glob("*1.h5")]
        super().__init__(h5_files)

        self.pol_freq = pol_freq
        self.max_baseline_distance = max_baseline_distance
        self.max_date_apart = max_date_apart

        self.init_pairs_baseline_date()

        if not self.config_save.exists() or self.overwrite:
            self.config_save.parent.mkdir(parents=True, exist_ok=True)
            with self.config_save.open("wb") as f:
                pickle.dump(self, f)

    # ------------------------------------------------------------------
    # Standardized path getters
    # ------------------------------------------------------------------

    @staticmethod
    def _date_str(scan_or_date):
        """Return YYYYMMDD string from a Scan-like object or a string."""
        return getattr(scan_or_date, "date_str", str(scan_or_date))

    def get_processor_config_path(self):
        return self.out_folder / "processor.pkl"

    def get_path_folder(self, path):
        return self.out_folder / Path(path)

    def get_stack_folder(self, path):
        return self.get_path_folder(path) / "stack"

    def get_ref_geom_folder(self, path):
        return self.get_path_folder(path) / "ref_geom"

    def get_coarse_offsets_folder(self, path):
        return self.get_path_folder(path) / "coarse_offsets"

    def get_coarse_offset_folder(self, path, date):
        return self.get_coarse_offsets_folder(path) / self._date_str(date)

    def get_geo2rdr_folder(self, path, date):
        return (
            self.get_coarse_offset_folder(path, date)
            / "geo2rdr"
            / f"freq{self.freq}"
        )

    def get_coarse_offset_file(self, path, date, component):
        if component not in {"azimuth", "range"}:
            raise ValueError(
                "component must be either 'azimuth' or 'range'"
            )
        return self.get_geo2rdr_folder(path, date) / f"{component}.off"

    def get_stack_slc(self, path, date):
        date_str = self._date_str(date)
        return self.get_stack_folder(path) / f"{date_str}.slc"

    def get_pair_name(self, ref, sec):
        return f"{self._date_str(ref)}_{self._date_str(sec)}"

    def get_dense_pairwise_folder(self, path):
        return self.get_path_folder(path) / "dense_pairwise"

    def get_dense_pair_folder(self, path, ref, sec):
        return (
            self.get_dense_pairwise_folder(path)
            / self.get_pair_name(ref, sec)
        )

    def get_dense_offsets_file(self, path, ref, sec):
        return self.get_dense_pair_folder(path, ref, sec) / "dense_offsets"

    def get_inverted_offsets_folder(self, path):
        return self.get_path_folder(path) / "inverted_offsets"

    def get_inverted_offset_folder(self, path, date):
        return self.get_inverted_offsets_folder(path) / self._date_str(date)

    def get_rubber_folder(self, path):
        return self.get_path_folder(path) / "rubber"

    def get_rubber_date_folder(self, path, date):
        return self.get_rubber_folder(path) / self._date_str(date)

    def get_rubber_pol_folder(self, path, date):
        return (
            self.get_rubber_date_folder(path, date)
            / f"freq{self.freq}"
            / self.pol
        )

    def get_h5_stack_folder(self, path):
        return self.get_path_folder(path) / "h5_stack"

    def get_h5_stack_date_folder(self, path, date):
        return self.get_h5_stack_folder(path) / self._date_str(date)

    def get_dense_offsets_link_folder(self, path):
        return (
            self.get_path_folder(path)
            / "dense_offsets"
            / f"freq{self.freq}"
            / self.pol
        )

    def get_rubbersheet_offsets_folder(self, path):
        return self.get_path_folder(path) / "rubbersheet_offsets"

    def get_merged_folder(self, path):
        return self.get_path_folder(path) / "merged"

    def get_merged_date_folder(self, path, date):
        return self.get_merged_folder(path) / self._date_str(date)

    def get_merged_slc(self, path, date):
        date_str = self._date_str(date)
        return self.get_merged_date_folder(path, date) / f"{date_str}.slc"

    def get_aligned_slc(self, path, date):
        """Alias for the final aligned SLC used by downstream workflows."""
        return self.get_merged_slc(path, date)

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def init_pairs_baseline_date(self):
        self.dense_pairs = {}

        for path, scans in self.scan_by_id.items():
            path_pairs = []

            for scan_a, scan_b in combinations(scans, 2):
                b = baseline(
                    scan_a.path_str,
                    scan_b.path_str,
                    freq=self.freq,
                )
                days_apart = abs((scan_a.date - scan_b.date).days)

                if self.debug:
                    print(
                        f"{self.get_pair_name(scan_a, scan_b)} "
                        f"{b['B']:12.2f} "
                        f"{b['Bperp']:12.2f} "
                        f"{b['Bpar']:14.2f} "
                        f"{b['Balong']:12.2f}"
                        f"{days_apart:12.2f}"
                    )

                baseline_check = b["B"] <= self.max_baseline_distance
                date_check = days_apart <= self.max_date_apart

                if baseline_check and date_check:
                    path_pairs.append((scan_a, scan_b))

            self.dense_pairs[path] = path_pairs

    def iter_path_dense_pairs(self, path):
        yield from self.dense_pairs[path]

    # ------------------------------------------------------------------
    # Coarse registration
    # ------------------------------------------------------------------

    def coarse_register_scans(self, rdr2geo_cfg=rdr2geo_cfg):
        for path in self.scan_by_id:
            main_ref = self.ref_scans[path]

            path_folder = self.get_path_folder(path)
            stack_folder = self.get_stack_folder(path)
            ref_geom_folder = self.get_ref_geom_folder(path)
            coarse_offsets_folder = self.get_coarse_offsets_folder(path)

            path_folder.mkdir(parents=True, exist_ok=True)
            stack_folder.mkdir(parents=True, exist_ok=True)
            coarse_offsets_folder.mkdir(parents=True, exist_ok=True)

            reference_slc = self.get_stack_slc(path, main_ref)

            if not ref_geom_folder.exists() or self.overwrite:
                ref_geom_folder.mkdir(parents=True, exist_ok=True)
                extract_ref_topo(
                    main_ref.path_str,
                    self.dem_file,
                    rdr2geo_cfg,
                    ref_geom_folder,
                )

            if not reference_slc.exists() or self.overwrite:
                copy_raster(
                    main_ref.path_str,
                    self.freq,
                    self.pol,
                    1024,
                    reference_slc,
                    file_type="ENVI",
                )

            pairs = {}

            for ref, sec in self.iter_path_ref_sec(path):
                coarse_offset_folder = self.get_coarse_offset_folder(
                    path,
                    sec,
                )

                pairs[(ref.date_str, sec.date_str)] = {
                    "azimuth": self.get_coarse_offset_file(
                        path,
                        sec,
                        "azimuth",
                    ),
                    "range": self.get_coarse_offset_file(
                        path,
                        sec,
                        "range",
                    ),
                }

                if not coarse_offset_folder.exists() or self.overwrite:
                    coarse_offset_folder.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    align_secondary(
                        sec.path_str,
                        self.dem_file,
                        ref_geom_folder,
                        coarse_offset_folder,
                    )

                sec_slc = self.get_stack_slc(path, sec)

                if not sec_slc.exists() or self.overwrite:
                    print(
                        f"calculating coarse reg for "
                        f"{path} {sec.date_str}"
                    )

                    resample(
                        stack_folder,
                        self.get_geo2rdr_folder(path, sec),
                        main_ref.path_str,
                        sec.path_str,
                        sec.date_str,
                    )

    # ------------------------------------------------------------------
    # Dense offsets
    # ------------------------------------------------------------------

    def dense_offset_pairs(self, dense_cfg=dense_cfg):
        for path in self.scan_by_id:
            dense_folder = self.get_dense_pairwise_folder(path)
            dense_folder.mkdir(parents=True, exist_ok=True)

            for ref, sec in self.iter_path_dense_pairs(path):
                pair_name = self.get_pair_name(ref, sec)
                out_dir = self.get_dense_pair_folder(path, ref, sec)

                if not out_dir.exists() or self.overwrite:
                    print(
                        f"calculating dense offset {path} {pair_name}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    dense_offset_coregistered(
                        self.get_stack_slc(path, ref),
                        self.get_stack_slc(path, sec),
                        out_dir,
                        dense_cfg=dense_cfg,
                        gpu_id=0,
                    )

    def invert_dense_offset_pairs(self):
        for path in self.scan_by_id:
            inverted_offsets_folder = self.get_inverted_offsets_folder(path)

            # Preserve original behavior: debug=True forces this block.
            if not inverted_offsets_folder.exists() or self.debug:
                inverted_offsets_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pairs = {}

                for ref, sec in self.iter_path_dense_pairs(path):
                    pairs[(ref.date_str, sec.date_str)] = (
                        self.get_dense_offsets_file(path, ref, sec)
                    )

                if pairs:
                    invert_offsets_isce3(
                        pair_offsets=pairs,
                        output_dir=str(inverted_offsets_folder),
                        reference_date=self.ref_scans[path].date_str,
                    )

    # ------------------------------------------------------------------
    # Rubbersheet
    # ------------------------------------------------------------------

    def rubbersheet(self, rubbersheet_cfg=rubbersheet_cfg):
        for path in self.scan_by_id:
            path_folder = self.get_path_folder(path)
            h5_stack_folder = self.get_h5_stack_folder(path)
            rubber_folder = self.get_rubber_folder(path)

            h5_stack_folder.mkdir(parents=True, exist_ok=True)
            rubber_folder.mkdir(parents=True, exist_ok=True)

            main_ref = self.ref_scans[path]

            for ref, sec in self.iter_path_ref_sec(path):
                coarse_offset_folder = self.get_coarse_offset_folder(
                    path,
                    sec,
                )
                out_dir = self.get_h5_stack_date_folder(path, sec)
                rubber_date_folder = self.get_rubber_date_folder(
                    path,
                    sec,
                )

                if not rubber_date_folder.exists() or self.overwrite:
                    print(
                        f"calculating rubbersheet "
                        f"{path} {sec.date_str}"
                    )

                    cfg["processing"]["rubbersheet"] = rubbersheet_cfg

                    inverted_offset_folder = (
                        self.get_inverted_offset_folder(path, sec)
                    )
                    dense_pair_folder = self.get_dense_pair_folder(
                        path,
                        ref,
                        sec,
                    )
                    link_to = self.get_dense_offsets_link_folder(path)

                    relative_symlink_contents(
                        dense_pair_folder,
                        link_to,
                    )
                    relative_symlink_contents(
                        inverted_offset_folder,
                        link_to,
                    )

                    cfg["dynamic_ancillary_file_group"]["dem_file"] = str(
                        self.dem_file
                    )

                    set_nested(
                        cfg,
                        [
                            "input_file_group",
                            "reference_rslc_file",
                        ],
                        main_ref.path_str,
                    )
                    set_nested(
                        cfg,
                        [
                            "product_path_group",
                            "sas_output_file",
                        ],
                        out_dir,
                    )
                    set_nested(
                        cfg,
                        [
                            "product_path_group",
                            "scratch_path",
                        ],
                        path_folder,
                    )

                    cfg["processing"]["rubbersheet"][
                        "geo2rdr_offsets_path"
                    ] = coarse_offset_folder

                    cfg["processing"]["rubbersheet"][
                        "dense_offsets_path"
                    ] = path_folder

                    cfg["processing"]["input_subset"][
                        "list_of_frequencies"
                    ] = {
                        self.freq: [self.pol]
                    }

                    create_minimal_rifg_for_rubbersheet(
                        out_dir,
                        cfg,
                    )
                    run_rubbersheet_with_interpolation(
                        cfg,
                        out_dir,
                    )

                    shutil.move(
                        self.get_rubbersheet_offsets_folder(path),
                        rubber_date_folder,
                    )

    # ------------------------------------------------------------------
    # Final aligned SLC stack
    # ------------------------------------------------------------------

    def resample_slcs(self):
        for path in self.scan_by_id:
            merged_folder = self.get_merged_folder(path)
            merged_folder.mkdir(parents=True, exist_ok=True)

            main_ref = self.ref_scans[path]
            ref_slc_folder = self.get_merged_date_folder(
                path,
                main_ref,
            )
            reference_slc = self.get_merged_slc(
                path,
                main_ref,
            )

            if not reference_slc.exists() or self.overwrite:
                print(f"copying reference slc for {path}")
                ref_slc_folder.mkdir(parents=True, exist_ok=True)

                copy_raster(
                    main_ref.path_str,
                    self.freq,
                    self.pol,
                    1024,
                    reference_slc,
                    file_type="ENVI",
                )

            for _, sec in self.iter_path_ref_sec(path):
                print(
                    f"calculating final offset for "
                    f"{path} {sec.date_str}"
                )

                rubber_pol_folder = self.get_rubber_pol_folder(
                    path,
                    sec,
                )
                pairs_out = self.get_merged_date_folder(
                    path,
                    sec,
                )

                if not pairs_out.exists() or self.overwrite:
                    print(
                        f"resampling {path} {sec.date_str}"
                    )

                    resample(
                        pairs_out,
                        rubber_pol_folder,
                        main_ref.path_str,
                        sec.path_str,
                        out_tag=sec.date_str,
                    )
