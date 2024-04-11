import contextlib
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Optional, Union

import isce3
import numpy as np
from isce3.ext.isce3.unwrap import _snaphu_unwrap


@dataclass(frozen=True)
class TopoCostParams:
    r"""Configuration parameters for SNAPHU "topo" cost mode

    Unwrapped phase is modelled as a topographic SAR interferometry signal
    in additive phase noise.

    Backscatter brightness is modelled according to the scattering model:

    .. math::

        \sigma^0 = C * \left( kds * \cos \theta_{inc} +
                \cos ^n (2 \theta_{inc}) \right) * \cos \theta_{inc}

    where :math:`C` is a scaling factor, :math:`kds` is the ratio of diffuse to
    specular scattering, :math:`\theta_{inc}` is the local incidence angle, and
    :math:`n` (input parameter `specular_exp`) is the power to which the
    specular cosine term is raised.

    Attributes
    ----------
    bperp : float
        Perpendicular baseline length, in meters. If the value is negative,
        increasing phase implies increasing topographic height.
    near_range : float
        Slant range from platform to the first range bin, in meters.
    dr, da : float
        Range & azimuth bin spacing after any multi-looking, in meters.
    range_res, az_res : float
        Single-look range & azimuth resolution, in meters.
    wavelength : float
        Wavelength, in meters.
    transmit_mode : {"pingpong", "repeat_pass", "single_antenna_transmit"}
        Radar transmit mode. 'pingpong' and 'repeat_pass' modes indicate that
        both antennas both transmitted and received. Both modes have the same
        effect in the algorithm. 'single_antenna_transmit' indicates that a
        single antenna was used to transmit while both antennas received. In
        this mode, the baseline is effectively halved.
    altitude : float
        Platform altitude relative to the Earth's surface, in meters.
    earth_radius : float, optional
        Local Earth radius, in meters. A spherical-Earth model is used.
        (default: 6378000.0)
    kds : float, optional
        Ratio of diffuse to specular scattering. (default: 0.02)
    specular_exp : float, optional
        Power specular scattering component. Larger values imply a sharper peak
        for specular scatter. (default: 8.0)
    dzr_crit_factor : float, optional
        Multiplicative factor applied to diffuse scatter term in evaluating
        crossover point between diffuse and specular scatter in terms of range
        slope. (default: 2.0)
    shadow : bool, optional
        Allow discontinuities from shadowing? If this is disabled, the minimum
        topographic slope estimated from mean backscatter intensity is clipped
        to the value of `dz_ei_min`. (default: False)
    dz_ei_min : float, optional
        Minimum slope expected in the absence of layover, in meters per
        slant-range pixel. (default: -4.0)
    lay_width : int, optional
        Width of window (number of pixels) for summing layover brightness.
        (default: 16)
    lay_min_ei : float, optional
        Threshold brightness (normalized) for assuming layover. (default: 1.25)
    slope_ratio_factor : float, optional
        Multiplicative factor applied to kds in order to get ratio of slopes for
        linearized scattering model. The term improves agreement of the
        piecewise-linear model with the cosine model near the transition point
        (dzrcrit) at the expense of poorer agreement at very large slopes.
        (default: 1.18)
    sigsq_ei : float, optional
        Variance of range slopes due to uncertainties in slope estimation from
        brightness, in (meters/pixel)^2. (default: 100.0)
    drho : float, optional
        Step size for calculating lookup table of maximum layover slope based on
        measured correlation. (default: 0.005)
    dz_lay_peak : float, optional
        Layover peak location, in meters/pixel. (default: -2.0)
    azdz_factor : float, optional
        Factor applied to range layover probability density to get azimuth
        layover probability density. (default: 0.99)
    dz_ei_factor : float, optional
        Factor applied to slope expected from brightness without layover. Can
        account for underestimation of brightness from averaging with
        neighboring dark pixels when despeckling. (default: 4.0)
    dz_ei_weight : float, optional
        Weight applied to slope expected from brightness without layover. Must
        be between zero and one. Can reduce influence of intensity on
        non-layover slope. This is useful if there are lots of non-topographic
        variations in brightness (i.e. changes in surface reflectivity).
        (default: 0.5)
    dz_lay_factor : float, optional
        Factor applied to slope expected from brightness with layover. Can
        account for underestimation of brightness from averaging with
        neighboring dark pixels when despeckling. (default: 1.0)
    lay_const : float, optional
        Ratio of layover probability density to peak probability density for
        non-layover slopes expected. (default: 0.9)
    lay_falloff_const : float, optional
        Factor applied to slope variance for non-layover to get falloff of
        probability density after the  upper layover slope limit has been
        exceeded. (default: 2.0)
    sigsq_lay_factor : float, optional
        Fraction of (ambiguity height)^2 to use for slope variance in the
        presence of layover. (default: 0.1)
    krow_ei, kcol_ei : int, optional
        Number of rows & columns to use in sliding average window used for
        normalizing intensity values. (default: 65, 257)
    init_dzr : float, optional
        Initial value of range slope for dzrcrit numerical solution, in
        meters/pixel. (default: 2048.0)
    init_dz_step : float, optional
        Initial range slope step size in dzrhomax numerical solution, in
        meters/pixel. (default: 100.0)
    cost_scale_ambig_ht : float, optional
        Ambiguity height for auto-scaling the `SolverParams.cost_scale`
        parameter to equal 100. This is the amount of height change, in meters,
        that results in a :math:`2 \pi` change in the interferometric phase. The
        cost scale is automatically adjusted to be inversely proportional to the
        midswath ambiguity height. (default: 80.0)
    dnom_inc_angle : float, optional
        Step size, in radians, for dzrhomax lookup table. The index is on the
        flat-earth incidence angle; this is the sample spacing in the table.
        (default: 0.01)
    kpar_dpsi, kperp_dpsi : int, optional
        Number of pixels in sliding window used for averaging wrapped phase
        gradients to get mean non-layover slope, in directions parallel and
        perpendicular to the examined phase difference. (default: 7, 7)
    """

    bperp: float
    near_range: float
    dr: float
    da: float
    range_res: float
    az_res: float
    wavelength: float
    transmit_mode: str
    altitude: float
    earth_radius: float = 6_378_000.0
    kds: float = 0.02
    specular_exp: float = 8.0
    dzr_crit_factor: float = 2.0
    shadow: bool = False
    dz_ei_min: float = -4.0
    lay_width: int = 16
    lay_min_ei: float = 1.25
    slope_ratio_factor: float = 1.18
    sigsq_ei: float = 100.0
    drho: float = 0.005
    dz_lay_peak: float = -2.0
    azdz_factor: float = 0.99
    dz_ei_factor: float = 4.0
    dz_ei_weight: float = 0.5
    dz_lay_factor: float = 1.0
    lay_const: float = 0.9
    lay_falloff_const: float = 2.0
    sigsq_lay_factor: float = 0.1
    krow_ei: int = 65
    kcol_ei: int = 257
    init_dzr: float = 2048.0
    init_dz_step: float = 100.0
    cost_scale_ambig_ht: float = 80.0
    dnom_inc_angle: float = 0.01
    kpar_dpsi: int = 7
    kperp_dpsi: int = 7

    def tostring(self):
        """Convert to string in SNAPHU config file format."""

        def parse_transmit_mode(s):
            if s == "pingpong":
                return "PINGPONG"
            if s == "repeat_pass":
                return "REPEATPASS"
            if s == "single_antenna_transmit":
                return "SINGLEANTENNATRANSMIT"
            raise ValueError(f"invalid transmit mode '{s}'")

        s = ""
        s += f"BPERP {self.bperp}\n"
        s += f"NEARRANGE {self.near_range}\n"
        s += f"DR {self.dr}\n"
        s += f"DA {self.da}\n"
        s += f"RANGERES {self.range_res}\n"
        s += f"AZRES {self.az_res}\n"
        s += f"LAMBDA {self.wavelength}\n"
        s += f"TRANSMITMODE {parse_transmit_mode(self.transmit_mode)}\n"
        s += f"ALTITUDE {self.altitude}\n"
        s += f"EARTHRADIUS {self.earth_radius}\n"
        s += f"KDS {self.kds}\n"
        s += f"SPECULAREXP {self.specular_exp}\n"
        s += f"DZRCRITFACTOR {self.dzr_crit_factor}\n"
        s += f"SHADOW {self.shadow}\n"
        s += f"DZEIMIN {self.dz_ei_min}\n"
        s += f"LAYWIDTH {self.lay_width}\n"
        s += f"LAYMINEI {self.lay_min_ei}\n"
        s += f"SLOPERATIOFACTOR {self.slope_ratio_factor}\n"
        s += f"SIGSQEI {self.sigsq_ei}\n"
        s += f"DRHO {self.drho}\n"
        s += f"DZLAYPEAK {self.dz_lay_peak}\n"
        s += f"AZDZFACTOR {self.azdz_factor}\n"
        s += f"DZEIFACTOR {self.dz_ei_factor}\n"
        s += f"DZEIWEIGHT {self.dz_ei_weight}\n"
        s += f"DZLAYFACTOR {self.dz_lay_factor}\n"
        s += f"LAYCONST {self.lay_const}\n"
        s += f"LAYFALLOFFCONST {self.lay_falloff_const}\n"
        s += f"SIGSQLAYFACTOR {self.sigsq_lay_factor}\n"
        s += f"KROWEI {self.krow_ei}\n"
        s += f"KCOLEI {self.kcol_ei}\n"
        s += f"INITDZR {self.init_dzr}\n"
        s += f"INITDZSTEP {self.init_dz_step}\n"
        s += f"COSTSCALEAMBIGHT {self.cost_scale_ambig_ht}\n"
        s += f"DNOMINCANGLE {self.dnom_inc_angle}\n"
        s += f"KPARDPSI {self.kpar_dpsi}\n"
        s += f"KPERPDPSI {self.kperp_dpsi}\n"
        return s


@dataclass(frozen=True)
class DefoCostParams:
    """Configuration parameters for SNAPHU "defo" cost mode

    Unwrapped phase is modelled as a surface deformation signature in additive
    phase noise.

    Attributes
    ----------
    azdz_factor : float, optional
        Factor applied to range discontinuity probability density to get
        corresponding value for azimuth. (default: 1.0)
    defo_max : float, optional
        Maximum phase discontinuity, in units of cycles (of 2*pi). If abrupt
        phase discontinuities are not expected, this parameter can be set to
        zero. (default: 1.2)
    sigsq_corr : float, optional
        Phase variance, in cycles^2, reflecting uncertainty in measurement of
        actual statistical correlation. (default: 0.05)
    defo_const : float, optional
        Ratio of phase discontinuity probability density to peak probability
        density expected for discontinuity-possible pixel differences. A value
        of 1 means zero cost for discontinuity, 0 means infinite cost.
        (default: 0.9)
    lay_falloff_const : float, optional
        Factor applied to slope variance for non-layover to get falloff of
        probability density after the  upper layover slope limit has been
        exceeded. (default: 2.0)
    kpar_dpsi, kperp_dpsi : int, optional
        Number of pixels in sliding window used for averaging wrapped phase
        gradients to get mean non-layover slope, in directions parallel and
        perpendicular to the examined phase difference. (default: 7, 7)
    """

    azdz_factor: float = 1.0
    defo_max: float = 1.2
    sigsq_corr: float = 0.05
    defo_const: float = 0.9
    lay_falloff_const: float = 2.0
    kpar_dpsi: int = 7
    kperp_dpsi: int = 7

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"DEFOAZDZFACTOR {self.azdz_factor}\n"
        s += f"DEFOMAX_CYCLE {self.defo_max}\n"
        s += f"SIGSQCORR {self.sigsq_corr}\n"
        s += f"DEFOCONST {self.defo_const}\n"
        s += f"LAYFALLOFFCONST {self.lay_falloff_const}\n"
        s += f"KPARDPSI {self.kpar_dpsi}\n"
        s += f"KPERPDPSI {self.kperp_dpsi}\n"
        return s


@dataclass(frozen=True)
class SmoothCostParams:
    """Configuration parameters for SNAPHU "smooth" cost mode

    Attributes
    ----------
    kpar_dpsi, kperp_dpsi : int, optional
        Number of pixels in sliding window used for averaging wrapped phase
        gradients to get mean non-layover slope, in directions parallel and
        perpendicular to the examined phase difference. (default: 7, 7)
    """

    kpar_dpsi: int = 7
    kperp_dpsi: int = 7

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"KPARDPSI {self.kpar_dpsi}\n"
        s += f"KPERPDPSI {self.kperp_dpsi}\n"
        return s


@dataclass(frozen=True)
class PNormCostParams:
    r"""Configuration parameters for SNAPHU "p-norm" cost mode

    In this mode, the minimization objective is the :math:`L^p` norm of the
    difference between the unwrapped and wrapped phase gradients.

    .. math:: cost = \sum_i \left| \Delta \phi_i - \Delta \psi_i \right| ^ p

    Attributes
    ----------
    p : float, optional
        Lp norm exponent. Must be nonnegative. (default: 0.0)
    bidir : bool, optional
        If True, bidirectional Lp costs are used. This implies that the scalar
        weight of an Lp arc may be different depending on the direction of net
        flow on the arc. If False, the weight is the same regardless of arc
        direction. (default: True)
    """

    p: float = 0.0
    bidir: bool = True

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"PLPN {self.p}\n"
        s += f"BIDIRLPN {self.bidir}\n"
        return s


@dataclass(frozen=True)
class TilingParams:
    """Configuration parameters affecting scene tiling and parallel processing.

    Attributes
    ----------
    nproc : int, optional
        Maximum number of child processes to spawn for parallel tile unwrapping.
        If nproc is less than 1, use all available processors. (default: 1)
    tile_nrows, tile_ncols : int, optional
        Number of tiles along the row/column directions. If `tile_nrows` and
        `tile_ncols` are both 1, the interferogram is unwrapped as a single
        tile. (default: 1, 1)
    row_overlap, col_overlap : int, optional
        Overlap, in number of rows/columns, between neighboring tiles.
        (default: 0)
    tile_cost_thresh : int, optional
        Cost threshold to use for determining boundaries of reliable regions.
        Larger cost threshold implies smaller regions (safer, but more expensive
        computationally). (default: 500)
    min_region_size : int, optional
        Minimum size of a reliable region in tile mode, in pixels. (default: 100)
    tile_edge_weight : float, optional
        Extra weight applied to secondary arcs on tile edges. (default: 2.5)
    secondary_arc_flow_max : int, optional
        Maximum flow magnitude whose cost will be stored in the secondary cost
        lookup table. Secondary costs larger than this will be approximated by a
        quadratic function. (default: 8)
    single_tile_reoptimize : bool, optional
        If True, re-optimize as a single tile after using tile mode for
        initialization. This is equivalent to unwrapping with multiple tiles,
        then using the unwrapped output as the input to a new, single-tile run
        of snaphu to make iterative improvements to the solution. This may
        improve speed compared to a single single-tile run. (default: False)
    """

    nproc: int = 1
    tile_nrows: int = 1
    tile_ncols: int = 1
    row_overlap: int = 0
    col_overlap: int = 0
    tile_cost_thresh: int = 500
    min_region_size: int = 100
    tile_edge_weight: float = 2.5
    secondary_arc_flow_max: int = 8
    single_tile_reoptimize: bool = False

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        if self.nproc < 1:
            nproc = os.cpu_count() or 1
            s += f"NPROC {nproc}\n"
        else:
            s += f"NPROC {self.nproc}\n"
        s += f"NTILEROW {self.tile_nrows}\n"
        s += f"NTILECOL {self.tile_ncols}\n"
        s += f"ROWOVRLP {self.row_overlap}\n"
        s += f"COLOVRLP {self.col_overlap}\n"
        s += f"TILECOSTTHRESH {self.tile_cost_thresh}\n"
        s += f"MINREGIONSIZE {self.min_region_size}\n"
        s += f"TILEEDGEWEIGHT {self.tile_edge_weight}\n"
        s += f"SCNDRYARCFLOWMAX {self.secondary_arc_flow_max}\n"
        s += f"SINGLETILEREOPTIMIZE {self.single_tile_reoptimize}\n"

        # Don't remove temporary files for each tile since they may be useful
        # for debugging. If the scratch directory is cleaned up, they'll be
        # removed as well.
        s += "RMTMPTILE FALSE\n"

        return s


@dataclass(frozen=True)
class SolverParams:
    """Configuration parameters used by the network initialization and nonlinear
    network flow solver algorithms.

    Attributes
    ----------
    max_flow_inc : int, optional
        Maximum flow increment. (default: 4)
    init_max_flow : int, optional
        Maximum flow to allow in initialization. If this is zero, then the
        maximum is calculated automatically from the statistical cost functions.
        To disable, set it to a large value like 9999, but do not overflow the
        long integer data type. (default: 9999)
    arc_max_flow_const : int, optional
        Constant to add to maximum flow expected from statistical cost functions
        for automatically determining initial maximum flow. (default: 3)
    threshold : float, optional
        Threshold precision for iterative numerical calculations.
        (default: 0.001)
    max_cost : float, optional
        Maximum cost allowed for scalar MST costs and for estimating the number
        of buckets needed for the solver routine. (default: 1000.0)
    cost_scale : float, optional
        Cost scaling factor applied to floating-point costs before quantization
        to integer costs. (default: 100.0)
    n_cycle : int, optional
        Integer spacing that represents one unit of flow (one cycle of phase)
        when storing costs as short integers. (default: 200)
    max_new_node_const : float, optional
        Fraction of total number of nodes to add in each tree expansion phase of
        the solver algorithm. (default: 0.0008)
    max_n_flow_cycles : float or None, optional
        Number of cycles to allow for a call to the solver with a specific flow
        increment delta and still consider that increment done. Ideally it would
        be zero, but scaling for different deltas may leave some negative cycles
        that won't affect the solution much. If None, this is automatically
        determined based on the size of the interferogram. (default: None)
    max_cycle_frac : float, optional
        Fraction of the number of pixels to use as the maximum number of cycles
        allowed for a specific flow increment if max_n_flow_cycles was None.
        (default: 0.00001)
    n_conn_node_min : int, optional
        Minimum number of connected nodes to consider for unwrapping. If masking
        separates the input data into disconnected sets of pixels, a source is
        selected for each connected set, provided that the number of nodes in
        the set is greater than n_conn_node_min. Must be nonnegative.
        (default: 0)
    n_major_prune : int, optional
        Number of major iterations between tree pruning operations. A smaller
        number causes pruning to occur more frequently. (default: 2000000000)
    prune_cost_thresh : int, optional
        Cost threshold for pruning the tree. A lower threshold prunes more
        aggressively. (default: 2000000000)
    """

    max_flow_inc: int = 4
    init_max_flow: int = 9999
    arc_max_flow_const: int = 3
    threshold: float = 0.001
    max_cost: float = 1000.0
    cost_scale: float = 100.0
    n_cycle: int = 200
    max_new_node_const: float = 0.0008
    max_n_flow_cycles: Optional[float] = None
    max_cycle_frac: float = 0.00001
    n_conn_node_min: int = 0
    n_major_prune: int = 2_000_000_000
    prune_cost_thresh: int = 2_000_000_000

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"MAXFLOW {self.max_flow_inc}\n"
        s += f"INITMAXFLOW {self.init_max_flow}\n"
        s += f"ARCMAXFLOWCONST {self.arc_max_flow_const}\n"
        s += f"THRESHOLD {self.threshold}\n"
        s += f"MAXCOST {self.max_cost}\n"
        s += f"COSTSCALE {self.cost_scale}\n"
        s += f"NSHORTCYCLE {self.n_cycle}\n"
        s += f"MAXNEWNODECONST {self.max_new_node_const}\n"
        if self.max_n_flow_cycles is not None:
            s += f"MAXNFLOWCYCLES {self.max_n_flow_cycles}\n"
        else:
            s += f"MAXCYCLEFRACTION {self.max_cycle_frac}\n"
        s += f"NCONNNODEMIN {self.n_conn_node_min}\n"
        s += f"NMAJORPRUNE {self.n_major_prune}\n"
        s += f"PRUNECOSTTHRESH {self.prune_cost_thresh}\n"
        return s


@dataclass(frozen=True)
class ConnCompParams:
    """Configuration parameters affecting the generation of connected component
    labels.

    Attributes
    ----------
    min_frac_area : float, optional
        Minimum size of a single connected component, as a fraction of the total
        number of pixels in the tile. (default: 0.01)
    cost_thresh : int, optional
        Cost threshold for connected components. Higher threshold will give
        smaller connected components. (default: 300)
    max_ncomps : int, optional
        Maximum number of connected components per tile. (default: 32)
    """

    min_frac_area: float = 0.01
    cost_thresh: int = 300
    max_ncomps: int = 32

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"MINCONNCOMPFRAC {self.min_frac_area}\n"
        s += f"CONNCOMPTHRESH {self.cost_thresh}\n"
        s += f"MAXNCOMPS {self.max_ncomps}\n"
        return s


@dataclass(frozen=True)
class CorrBiasModelParams:
    r"""Model parameters for estimating bias in sample correlation magnitude
    expected for zero true correlation

    The multilooked correlation magnitude of the interferometric pair
    :math:`z_1` and :math:`z_2` is commonly estimated as

    .. math::

        \rho = \left| \frac{ \sum_{i=1}^{N}{z_{1i} z_{2i} ^*} }
            { \sqrt{ \sum_{i=1}^{N}{ \left| z_{1i} \right| ^2 } }
            \sqrt{ \sum_{i=1}^{N}{ \left| z_{2i} \right| ^2 } } }  \right|

    where :math:`N` is the number of statistically independent looks. SNAPHU
    uses the estimated correlation coefficient to infer statistics of the
    interferometric phase.

    This estimator is biased with respect to the expected true correlation,
    particularly at lower correlation values. In order to compensate for this,
    SNAPHU models the expected biased correlation measure, given that true
    interferometric correlation is zero, as

    .. math:: \rho_0 = \frac{c_1}{N} + c_2

    where :math:`N` is the number of effective looks used to estimate the
    correlation and :math:`c_1` & :math:`c_2` are the model coefficients. This
    approximately matches the curves of Touzi et al [1]_.

    Attributes
    ----------
    c1, c2 : float, optional
        Correlation bias model parameters.
    min_corr_factor : float, optional
        Factor applied to expected minimum measured (biased) correlation
        coefficient. Values smaller than the threshold min_corr_factor * rho0
        are assumed to come from zero statistical correlation because of
        estimator bias. rho0 is the expected biased correlation measure if the
        true correlation is zero. (default: 1.25)

    References
    ----------
    .. [1] R. Touzi, A. Lopes, J. Bruniquel, and P. W. Vachon, "Coherence
       estimation for SAR imagery," IEEE Trans. Geosci. Remote Sens. 37, 135-149
       (1999).
    """

    c1: float = 1.3
    c2: float = 0.14
    min_corr_factor: float = 1.25

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"RHOSCONST1 {self.c1}\n"
        s += f"RHOSCONST2 {self.c2}\n"

        # This parameter has different names depending on which cost mode was
        # selected -- "RHOMINFACTOR" for "topo" mode, "DEFOTHRESHFACTOR" for
        # "defo" and "smooth" mode -- but the semantics & effect are the same.
        # We redundantly define both params here since it's harmless and avoids
        # introducing a dependency on cost mode.
        s += f"RHOMINFACTOR {self.min_corr_factor}\n"
        s += f"DEFOTHRESHFACTOR {self.min_corr_factor}\n"

        return s


@dataclass(frozen=True)
class PhaseStddevModelParams:
    r"""Model parameters for approximating phase standard deviation from
    correlation magnitude

    Interferometric phase standard deviation is modelled as

    .. math:: \sigma_{\phi} = \rho ^{ c_1 + c_2 * \log nlooks + c_3 * nlooks }

    where :math:`\rho` is the sample correlation magnitude and :math:`nlooks` is
    the effective number of looks. :math:`c_1`, :math:`c_2`, and :math:`c_3` are
    the model coefficients. This approximately matches the curves of [1]_.

    Attributes
    ----------
    c1, c2, c3 : float, optional
        Interferometric phase standard deviation model parameters.
    sigsq_min : int
        Minimum value of phase variance after quantization to integer values.
        Must be greater than zero to prevent division by zero. (default: 1)

    References
    ----------
    .. [1] J. S. Lee, K. W. Hoppel, S. A. Mango, and A. R. Miller, "Intensity
       and phase statistics of multilook polarimetric and interferometric SAR
       imagery," IEEE Trans. Geosci. Remote Sens. 32, 1017-1028 (1994).
    """

    c1: float = 0.4
    c2: float = 0.35
    c3: float = 0.06
    sigsq_min: int = 1

    def tostring(self):
        """Convert to string in SNAPHU config file format."""
        s = ""
        s += f"CSTD1 {self.c1}\n"
        s += f"CSTD2 {self.c2}\n"
        s += f"CSTD3 {self.c3}\n"
        s += f"SIGSQSHORTMIN {self.sigsq_min}\n"
        return s


@contextlib.contextmanager
def scratch_directory(d: Optional[os.PathLike] = None) -> pathlib.Path:
    """Context manager that creates a (possibly temporary) filesystem directory

    If the input is a path-like object, a directory will be created at the
    specified filesystem path if it did not already exist. The directory will
    persist after leaving the context manager scope.

    If the input is None, a temporary directory is created as though by
    `tempfile.TemporaryDirectory()`. Upon exiting the context manager scope, the
    directory and its contents are removed from the filesystem.

    Parameters
    ----------
    d : path-like or None, optional
        Scratch directory path. If None, a temporary directory is created.
        (default: None)

    Yields
    ------
    d : pathlib.Path
        Scratch directory path. If the input was None, the directory is removed
        from the filesystem upon exiting the context manager scope.
    """
    if d is None:
        try:
            d = tempfile.TemporaryDirectory()
            yield pathlib.Path(d.name)
        finally:
            d.cleanup()
    else:
        d = pathlib.Path(d)
        d.mkdir(parents=True, exist_ok=True)
        yield d


def to_flat_file(
    path: os.PathLike,
    raster: isce3.io.gdal.Raster,
    dtype: Optional[np.dtype] = None,
    batchsize: int = -1,
):
    """Write raster data to flat binary file.

    The output file is overwritten if it exists.

    Parameters
    ----------
    path : path-like
        Output filepath.
    raster : isce3.io.gdal.Raster
        Input raster.
    dtype : data-type or None, optional
        Output datatype. If None, use the input raster datatype. (default: None)
    batchsize : int, optional
        If this is a positive number, the data is copied serially in batches of
        this many rows to avoid holding the full dataset in memory at once.
        Otherwise, copy the full data array as a single batch. (default: -1)
    """
    if dtype is None:
        dtype = raster.data.dtype

    if batchsize < 1:
        batchsize = raster.length

    # Memory-map the output file.
    shape = (raster.length, raster.width)
    mmap = np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    # Write data in batches.
    for i0 in range(0, raster.length, batchsize):
        i1 = i0 + batchsize
        mmap[i0:i1] = raster.data[i0:i1]

    # Explicitly flush to disk instead of waiting for the memory map to be
    # garbage-collected.
    mmap.flush()


def from_flat_file(
    path: os.PathLike,
    raster: isce3.io.gdal.Raster,
    dtype: Optional[np.dtype] = None,
    batchsize: int = -1,
):
    """Read raster data from flat binary file.

    Parameters
    ----------
    path : path-like
        Input filepath.
    raster : isce3.io.gdal.Raster
        Output raster.
    dtype : data-type or None, optional
        Input file datatype. If None, assume the same as the output raster
        datatype. (default: None)
    batchsize : int, optional
        If this is a positive number, the data is copied serially in batches of
        this many rows to avoid holding the full dataset in memory at once.
        Otherwise, copy the full data array as a single batch. (default: -1)
    """
    if dtype is None:
        dtype = raster.data.dtype

    if batchsize < 1:
        batchsize = raster.length

    # Memory-map the input file.
    shape = (raster.length, raster.width)
    mmap = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    # Read data in batches.
    for i0 in range(0, raster.length, batchsize):
        i1 = i0 + batchsize
        raster.data[i0:i1] = mmap[i0:i1]


CostParams = Union[
    TopoCostParams, DefoCostParams, SmoothCostParams, PNormCostParams,
]
CostParams.__doc__ = """SNAPHU cost mode configuration parameters"""


def unwrap(
    unw: isce3.io.gdal.Raster,
    conncomp: isce3.io.gdal.Raster,
    igram: isce3.io.gdal.Raster,
    corr: isce3.io.gdal.Raster,
    nlooks: float,
    cost: str = "smooth",
    cost_params: Optional[CostParams] = None,
    init_method: str = "mcf",
    pwr: Optional[isce3.io.gdal.Raster] = None,
    mask: Optional[isce3.io.gdal.Raster] = None,
    unwest: Optional[isce3.io.gdal.Raster] = None,
    tiling_params: Optional[TilingParams] = None,
    solver_params: Optional[SolverParams] = None,
    conncomp_params: Optional[ConnCompParams] = None,
    corr_bias_model_params: Optional[CorrBiasModelParams] = None,
    phase_stddev_model_params: Optional[PhaseStddevModelParams] = None,
    scratchdir: Optional[os.PathLike] = None,
    debug: bool = False,
):
    r"""Performs 2-D phase unwrapping on an input interferogram using the SNAPHU
    algorithm.

    The algorithm attempts to estimate the unwrapped phase field by
    approximately solving a non-linear optimization problem using cost functions
    based on simple statistical models of the unwrapped phase gradients.

    The total cost is approximately minimized by applying a non-linear network
    flow solver based on the network simplex algorithm. An initial feasible
    solution is first computed before optimizing according to the specified cost
    mode.

    Different statistical cost functions may be applied depending on the
    application:
    - The "topo" cost mode generates cost functions for topographic SAR
    interferometry. The problem statistics are based on the assumption that the
    true unwrapped phase represents surface elevation. The input interferogram
    is assumed to be in radar (range-azimuth) coordinates for this mode.
    - The "defo" cost mode generates cost functions for deformation
    measurements. The problem statistics are based on the assumption that the
    true unwrapped phase represents surface displacement.
    - The "smooth" cost mode models the problem statistics based on the
    assumption that the true unwrapped phase represents a generic surface with
    no discontinuities.
    - The "p-norm" cost mode is not based on a statistical model but rather
    minimizes the :math:`L^p` norm of the difference between the unwrapped and
    wrapped phase gradients.

    The outputs include the unwrapped phase and a raster of connected component
    labels. Each connected component is a region of pixels in the solution that
    is believed to have been unwrapped in an internally self-consistent manner.
    Each distinct region is assigned a unique positive integer label. Pixels not
    belonging to any component are assigned a label of zero.

    The effective number of looks used to form the input correlation data must
    be provided in order to estimate interferometric phase statistics. The
    effective number of looks is an estimate of the number of statistically
    independent samples averaged in multilooked data, taking into account
    spatial correlation due to oversampling/filtering. It is approximately equal
    to

    .. math:: n_e = k_r k_a \frac{d_r d_a}{\rho_r \rho_a}

    where :math:`k_r` and :math:`k_a` are the number of looks in range and
    azimuth, :math:`d_r` and :math:`d_a` are the sample spacing in range and
    azimuth, and :math:`\rho_r` and :math:`\rho_a are the range and azimuth
    resolution.

    Unwrapping can be performed in tile mode to potentially speed up processing
    and make use of multiple processors in parallel, though this may result in
    processing artifacts at tile boundaries. The interferogram is partitioned
    into rectangular tiles, each of which is unwrapped independently before
    reassembly. The default behavior is to unwrap the full interferogram as a
    single tile.

    .. warning:: Currently, if tile mode is used and any connected component
    crosses spans multiple tiles, the assigned connected component label may be
    inconsistent across tiles.

    Parameters
    ----------
    unw : isce3.io.gdal.Raster
        Output raster for unwrapped phase, in radians. Must have the same
        dimensions as the input interferogram and floating-point datatype.
    conncomp : isce3.io.gdal.Raster
        Output connected component labels. Must have the same dimensions as the
        input interferogram and integer datatype.
    igram : isce3.io.gdal.Raster
        Input interferogram. Must have complex datatype.
    corr : isce3.io.gdal.Raster
        Correlation magnitude, normalized to the interval [0, 1]. Must have the
        same dimensions as the input interferogram and floating-point datatype.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    cost : {"topo", "defo", "smooth", "p-norm"}, optional
        Statistical cost mode. (default: "smooth")
    cost_params : CostParams or None, optional
        Configuration parameters for the specified cost mode. This argument is
        required for "topo" mode and optional for all other modes. If None, the
        default configuration parameters are used. (default: None)
    init_method: {"mst", "mcf"}, optional
        Algorithm used for initialization of unwrapped phase gradients.
        Supported algorithms include Minimum Spanning Tree ("mst") and Minimum
        Cost Flow ("mcf"). (default: "mcf")
    pwr : isce3.io.gdal.Raster or None, optional
        Average intensity of the two SLCs, in linear units (not dB). Only used
        in "topo" cost mode. If None, interferogram magnitude is used as
        intensity. Must have the same dimensions as the input interferogram and
        floating-point datatype. (default: None)
    mask : isce3.io.gdal.Raster or None, optional
        Binary mask of valid pixels. Zeros in this raster indicate interferogram
        pixels that should be masked out. Must have the same dimensions as the
        input interferogram and GDT_Byte datatype. (default: None)
    unwest : isce3.io.gdal.Raster or None, optional
        Initial estimate of unwrapped phase, in radians. This can be used to
        provide a coarse unwrapped estimate to guide the algorithm. Must have
        the same dimensions as the input interferogram and floating-point
        datatype. (default: None)
    tiling_params : TilingParams or None, optional
        Configuration parameters affecting scene tiling and parallel processing.
        If None, the default configuration parameters are used. (default: None)
    solver_params : SolverParams or None, optional
        Configuration parameters used by the network initialization and
        nonlinear network flow solver algorithms. If None, the default
        configuration parameters are used. (default: None)
    conncomp_params : ConnCompParams or None, optional
        Configuration parameters affecting the generation of connected component
        labels. If None, the default configuration parameters are used.
        (default: None)
    corr_bias_model_params : CorrBiasModelParams or None, optional
        Model parameters for estimating bias in sample correlation magnitude
        expected for zero true correlation. If None, the default model
        parameters are used. (default: None)
    phase_stddev_model_params : PhaseStddevModelParams or None, optional
        Model parameters for approximating phase standard deviation from
        correlation magnitude. If None, the default model parameters are used.
        (default: None)
    scratchdir : path-like or None, optional
        Scratch directory where intermediate processing artifacts are written.
        If the specified directory does not exist, it will be created. If None,
        a temporary directory will be created and automatically removed from the
        filesystem at the end of processing. Otherwise, the directory and its
        contents will not be cleaned up. (default: None)
    debug : bool, optional
        Dump intermediate data arrays to scratch directory for debugging?
        (default: False)

    See Also
    --------
    isce3.unwrap.ICU : Branch-cut-based phase unwrapping
    isce3.unwrap.Phass : Minimum Cost Flow-based phase unwrapping

    References
    ----------
    .. [1] C. W. Chen and H. A. Zebker, "Network approaches to two-dimensional
       phase unwrapping: intractability and two new algorithms," Journal of the
       Optical Society of America A, vol. 17, pp. 401-414 (2000).
    .. [2] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping with
       use of statistical models for cost functions in nonlinear optimization,"
       Journal of the Optical Society of America A, vol. 18, pp. 338-351 (2001).
    .. [3] C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR
       interferograms: Statistical segmentation and generalized network models,"
       IEEE Transactions on Geoscience and Remote Sensing, vol. 40, pp.
       1709-1719 (2002).
    """
    # Verify input & output raster datatypes.
    if not np.issubdtype(unw.data.dtype, np.floating):
        raise TypeError("unw raster must have floating-point datatype")
    if not np.issubdtype(conncomp.data.dtype, np.integer):
        raise TypeError("conncomp raster must have integer datatype")
    if not np.issubdtype(igram.data.dtype, np.complexfloating):
        raise TypeError("igram raster must have complex datatype")
    if not np.issubdtype(corr.data.dtype, np.floating):
        raise TypeError("corr raster must have floating-point datatype")

    length, width = igram.length, igram.width

    # Check that raster dimensions are consistent.
    if (unw.length != length) or (unw.width != width):
        raise ValueError("unw raster dimensions must match interferogram")
    if (conncomp.length != length) or (conncomp.width != width):
        raise ValueError("conncomp raster dimensions must match interferogram")
    if (corr.length != length) or (corr.width != width):
        raise ValueError("corr raster dimensions must match interferogram")

    # Check specified number of effective looks.
    if nlooks < 1.0:
        raise ValueError("nlooks must be >= 1.0")

    # Generate a SNAPHU text configuration file to pass to the C++ code.
    configstr = ""
    configstr += f"LINELENGTH {width}\n"
    configstr += f"NCORRLOOKS {nlooks}\n"

    def cost_string():
        if cost == "topo":
            return "TOPO"
        if cost == "defo":
            return "DEFO"
        if cost == "smooth":
            return "SMOOTH"
        if cost == "p-norm":
            return "NOSTATCOSTS"
        raise ValueError(f"invalid cost mode '{cost}'")

    configstr += f"STATCOSTMODE {cost_string()}\n"

    def init_string():
        if init_method == "mst":
            return "MST"
        if init_method == "mcf":
            return "MCF"
        raise ValueError(f"invalid init method '{init_method}'")

    configstr += f"INITMETHOD {init_string()}\n"

    # Check cost mode-specific configuration params.
    if cost == "topo":
        # In "topo" mode, configuration params must be provided (there is no
        # default configuration).
        if not isinstance(cost_params, TopoCostParams):
            raise TypeError(
                "cost_params for 'topo' cost mode must be an "
                "instance of TopoCostParams"
            )
    elif cost == "defo":
        if cost_params is None:
            cost_params = DefoCostParams()
        if not isinstance(cost_params, DefoCostParams):
            raise TypeError("invalid cost_params for 'defo' cost mode")
    elif cost == "smooth":
        if cost_params is None:
            cost_params = SmoothCostParams()
        if not isinstance(cost_params, SmoothCostParams):
            raise TypeError("invalid cost_params for 'smooth' cost mode")
    elif cost == "p-norm":
        if cost_params is None:
            cost_params = PNormCostParams()
        if not isinstance(cost_params, PNormCostParams):
            raise TypeError("invalid cost_params for 'p-norm' cost mode")
    else:
        raise ValueError(f"invalid cost mode '{cost}'")

    configstr += cost_params.tostring()

    # Additional optional configuration parameters.
    if tiling_params is not None:
        configstr += tiling_params.tostring()
    if solver_params is not None:
        configstr += solver_params.tostring()
    if conncomp_params is not None:
        configstr += conncomp_params.tostring()

    # Curve-fitting coefficients.
    if corr_bias_model_params is not None:
        configstr += corr_bias_model_params.tostring()
    if phase_stddev_model_params is not None:
        configstr += phase_stddev_model_params.tostring()

    # Debug mode requires that a scratch directory is specified (otherwise, all
    # debug output would be automatically discarded anyway).
    if debug and (scratchdir is None):
        raise ValueError("scratchdir path must be specified if debug is True")

    # If no scratch directory was specified, make a temporary one. Otherwise,
    # create the directory if it doesn't already exist.
    with scratch_directory(scratchdir) as d:
        # SNAPHU expects flat binary data files, not GDAL rasters, as inputs &
        # outputs. Therefore, we create some intermediate files in the scratch
        # directory to pass to the backend code.

        # Output unwrapped data
        tmp_unw = d / "unw.f4"
        configstr += f"OUTFILE {tmp_unw.resolve()}\n"
        configstr += f"OUTFILEFORMAT FLOAT_DATA\n"

        # Output connected component labels
        tmp_conncomp = d / "conncomp.u4"
        configstr += f"CONNCOMPFILE {tmp_conncomp.resolve()}\n"
        configstr += f"CONNCOMPOUTTYPE UINT\n"

        # Input interferogram
        tmp_igram = d / "igram.c8"
        to_flat_file(tmp_igram, igram, dtype=np.complex64, batchsize=1024)
        configstr += f"INFILE {tmp_igram.resolve()}\n"
        configstr += f"INFILEFORMAT COMPLEX_DATA\n"

        # Input correlation magnitude
        tmp_corr = d / "corr.f4"
        to_flat_file(tmp_corr, corr, dtype=np.float32, batchsize=1024)
        configstr += f"CORRFILE {tmp_corr.resolve()}\n"
        configstr += f"CORRFILEFORMAT FLOAT_DATA\n"

        # Input SLC intensity data
        if pwr is not None:
            if cost != "topo":
                raise ValueError("SLC intensity data is only used in 'topo' mode")
            if not np.issubdtype(pwr.data.dtype, np.floating):
                raise TypeError("pwr raster must have floating-point datatype")
            if (pwr.length != length) or (pwr.width != width):
                raise ValueError("pwr raster dimensions must match interferogram")

            tmp_pwr = d / "pwr.f4"
            to_flat_file(tmp_pwr, pwr, dtype=np.float32, batchsize=1024)
            configstr += f"PWRFILE {tmp_pwr.resolve()}\n"
            configstr += f"AMPFILEFORMAT FLOAT_DATA\n"

        # Input data mask
        if mask is not None:
            if mask.datatype != isce3.io.gdal.GDT_Byte:
                raise TypeError("mask raster must have GDT_Byte datatype")
            if (mask.length != length) or (mask.width != width):
                raise ValueError("mask raster dimensions must match interferogram")

            tmp_mask = d / "mask.i1"
            to_flat_file(tmp_mask, mask, dtype=np.bool_, batchsize=1024)
            configstr += f"BYTEMASKFILE {tmp_mask.resolve()}\n"

        # Input unwrapped phase estimate
        if unwest is not None:
            if not np.issubdtype(unwest.data.dtype, np.floating):
                raise TypeError("unwest raster must have floating-point datatype")
            if (unwest.length != length) or (unwest.width != width):
                raise ValueError("unwest raster dimensions must match interferogram")

            tmp_unwest = d / "unwest.f4"
            to_flat_file(tmp_unwest, unwest, dtype=np.float32, batchsize=1024)
            configstr += f"ESTIMATEFILE {tmp_unwest.resolve()}\n"
            configstr += f"ESTFILEFORMAT FLOAT_DATA\n"

        configstr += f"DEBUG {debug}\n"

        # Ensure that debug outputs are written to the scratch directory.
        if debug:
            configstr += f"INITFILE {(d / 'snaphu.init')}\n"
            configstr += f"FLOWFILE {(d / 'snaphu.flow')}\n"
            configstr += f"ROWCOSTFILE {(d / 'snaphu.rowcost')}\n"
            configstr += f"COLCOSTFILE {(d / 'snaphu.colcost')}\n"
            configstr += f"CORRDUMPFILE {(d / 'snaphu.corr')}\n"

        # Write config params to file.
        configpath = d / "snaphu.conf"
        configpath.write_text(configstr)

        # Run SNAPHU.
        _snaphu_unwrap(str(configpath))

        # Copy output data to GDAL rasters.
        from_flat_file(tmp_unw, unw, dtype=np.float32, batchsize=1024)
        from_flat_file(tmp_conncomp, conncomp, dtype=np.uint32, batchsize=1024)
