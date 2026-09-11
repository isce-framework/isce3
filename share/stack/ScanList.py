

import numpy as np
import yaml
from pathlib import Path
import numpy as np
import yaml
import os
import re
from datetime import datetime
import isce3
from nisar.products.readers import RSLC


def baseline(ref_file, sec_file, freq="A"):
    ref = RSLC(hdf5file=ref_file)
    sec = RSLC(hdf5file=sec_file)

    rg1 = ref.getRadarGrid(freq)
    rg2 = sec.getRadarGrid(freq)

    orb1 = ref.getOrbit().copy()
    orb2 = sec.getOrbit().copy()

    # Make radar-grid times and orbit times use identical epochs
    orb1.update_reference_epoch(rg1.ref_epoch)
    orb2.update_reference_epoch(rg2.ref_epoch)

    # Center of reference radar grid
    t1 = rg1.sensing_mid
    r1 = rg1.mid_range

    ellipsoid = isce3.core.Ellipsoid()

    # Ground point observed at reference scene center
    llh = isce3.geometry.rdr2geo(
        t1,
        r1,
        orb1,
        rg1.lookside,
        doppler=0.0,
        wavelength=rg1.wavelength,
        ellipsoid=ellipsoid,
    )

    xyz = np.asarray(
        ellipsoid.lon_lat_to_xyz(llh),
        dtype=float
    )

    # Find where same ground point occurs in secondary acquisition
    zero_doppler = isce3.core.LUT2d()

    t2, r2 = isce3.geometry.geo2rdr(
        llh,
        ellipsoid,
        orb2,
        zero_doppler,
        rg2.wavelength,
        rg2.lookside,
    )

    p1, v1 = orb1.interpolate(t1)
    p2, v2 = orb2.interpolate(t2)

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    v1 = np.asarray(v1)

    B = p2 - p1

    # Unit LOS: reference satellite -> ground
    los = xyz - p1
    los /= np.linalg.norm(los)

    # Along-track direction
    along = v1 / np.linalg.norm(v1)

    # Cross-track direction perpendicular to LOS
    perp = np.cross(along, los)
    perp /= np.linalg.norm(perp)

    Bpar = np.dot(B, los)
    Bperp = np.dot(B, perp)
    Balong = np.dot(B, along)

    return {
        "B": np.linalg.norm(B),
        "Bperp": Bperp,
        "Bpar": Bpar,
        "Balong": Balong,
        "t_ref": t1,
        "t_sec": t2,
        "r_ref": r1,
        "r_sec": r2,
    }


def grep_date6(fn):
    m = re.search(r'20\d{6}', os.path.basename(fn))
    return m.group() if m else os.path.basename(fn)


def grep_date8(s):
    m = re.search(r'\d{8}', s)
    if m:
        date = m.group()
    return date


class Scan:
    def __init__(self, path):
        self.path = Path(path)
        self.path_str = str(path)
        self.stem = self.path.stem
        self.date_str = grep_date8(self.stem)
        self.date = datetime.strptime(self.date_str, "%Y%m%d")
        # self.scan_path = [f't{}_'  self.stem.split('_')[4:6]]
        parts = Path(self.stem).stem.split("_")
        self.cycle, self.track, self.direction, self.frame, \
            self.mode, self.polarization = int(parts[4]), int(parts[5]), \
            parts[6], int(parts[7]), parts[8], parts[9]
        self.scan_path_id = f't{self.track}_f{self.frame}_{self.direction}'

    def __str__(self):
        return f"Scan {self.stem}"

    def __repr__(self):
        return f"Scan {self.stem}"


def print_baseline_choices(reference, secondaries):
    ref = reference.path_str
    print("REFERENCE:", grep_date6(ref))
    for i in secondaries:
        sec = i.path_str
        print(i.scan_path_id)
        b = baseline(ref, sec)

        print(
            f"{grep_date6(sec):10s} "
            f"{b['B']:12.2f} "
            f"{b['Bperp']:12.2f} "
            f"{b['Bpar']:14.2f} "
            f"{b['Balong']:12.2f}"
        )


class ScanList:
    def __init__(self, files):
        self.files = files
        self.scans = [Scan(i) for i in files]
        self.scan_path_ids = np.unique([i.scan_path_id for i in self.scans])
        self.scan_by_id = {}
        self.ref_scans = {}
        for idi in self.scan_path_ids:
            scan_id_list = [i for i in self.scans if i.scan_path_id == idi]
            sort_id = np.argsort([i.date for i in scan_id_list])
            self.scan_by_id[idi] = [scan_id_list[i] for i in sort_id]
            self.ref_scans[idi] = scan_id_list[sort_id[0]]

    def iter_path_ref_sec(self, path):
        val = self.scan_by_id[path]
        for i in range(len(val) - 1):
            reference = val[i]
            secondary = val[i + 1]
            yield reference, secondary
