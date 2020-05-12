import h5py
import logging
import numpy as np
from pybind_isce3.core import LUT2d, DateTime
from nisar.types import complex32

log = logging.getLogger("SLCWriter")

class SLC(h5py.File):
    def __init__(self, *args, band="LSAR", product="RSLC", **kw):
        super().__init__(*args, **kw)
        self.band = band
        self.product = product
        self.root = self.create_group(f"/science/{band}/{product}")
        self.idroot = self.create_group(f"/science/{band}/identification")
        self.attrs["Conventions"] = np.string_("CF-1.7")
        self.attrs["contact"] = np.string_("nisarops@jpl.nasa.gov")
        self.attrs["institution"] = np.string_("NASA JPL")
        self.attrs["mission_name"] = np.string_("NISAR")
        self.attrs["reference_document"] = np.string_("TBD")
        self.attrs["title"] = np.string_("NISAR L1 RSLC Product")

    def create_dataset(self, *args, **kw):
        log.debug(f"Creating dataset {args[0]}")
        return super().create_dataset(*args, **kw)

    def create_group(self, *args, **kw):
        log.debug(f"Creating group {args[0]}")
        return super().create_group(*args, **kw)

    def set_doppler(self, dop: LUT2d, epoch: DateTime, frequency='A'):
        log.info(f"Saving Doppler for frequency {frequency}")
        g = self.root.require_group("metadata/processingInformation/parameters")
        # Actual LUT goes into a subdirectory, not created by serialization.
        name = f"frequency{frequency}"
        g.require_group(name)
        dop.save_to_h5(g, f"{name}/dopplerCentroid", epoch, "Hz")

    def swath(self, frequency="A") -> h5py.Group:
        return self.root.require_group(f"swaths/frequency{frequency}")

    def add_polarization(self, frequency="A", pol="HH"):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        g = self.swath(frequency)
        name = "listOfPolarizations"
        if name in g:
            pols = np.array(g[name])
            del g[name]
        else:
            pols = np.array([pol], dtype="S2")
        dset = g.create_dataset(name, data=pols)
        desc = f"List of polarization layers with frequecy {frequency}"
        dset.attrs["description"] = np.string_(desc)

    def create_image(self, frequency="A", pol="HH", **kw) -> h5py.Dataset:
        log.info(f"Creating SLC image for frequency={frequency} pol={pol}")
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        self.add_polarization(frequency, pol)
        kw.setdefault("dtype", complex32)
        dset = self.swath(frequency).create_dataset(pol, **kw)
        dset.attrs["description"] = np.string_(f"Focused SLC image ({pol})")
        dset.attrs["units"] = np.string_("DN")
        return dset
