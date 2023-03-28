#include "ICU.h"
#include <cstddef> // size_t
#include <isce3/io/Raster.h>

namespace py = pybind11;

using isce3::io::Raster;
using isce3::unwrap::icu::ICU;

void addbinding(py::class_<ICU> & pyICU)
{   pyICU.doc() = R"(
    Class for initializing ICU unwrapping algorithm
    
    Attributes
    ----------
    buffer_lines : int
         Number of buffer lines

    overlap_lines: int
         Number of overlapping lines
	 
    use_phase_grad_neut: bool
         Phase gradient neutron flag
	 
    use_intensity_neut : bool
         Intensity neutron flag
	 
    phase_grad_win_size : int
         Window size for phase gradient
	 
    neut_phase_grad_thr : float
         Threshold for phase gradient neutrons (radians)
	 
    neut_intensity_thr : float
         Threshold for intensity neutrons (sigma above mean)
	 
    neut_correlation_thr : float
         Maximum correlation for intensity neutrons
	 
    trees_number : int
         Number of realizations of  trees
	 
    max_branch_length : int
         Maximum branch length of trees
	 
    ratio_dxdy : float
         Ratio of pixel spacing in x and y directions
	 
    init_corr_thr : float
         Initial unwrap correlation threshold
	 
    max_corr_thr : float 
         Maximum unwrap correlation threshold
	 
    corr_incr_thr : float 
         Correlation threshold increments
	 
    min_cc_area : float
         Minimum connected component size fraction of tile area
	 
    num_bs_lines : int
         Number of bootstrap lines
	 
    min_overlap_area : int 
         Minimum bootstrap overlapping area
	 
    phase_var_thr : float
         Bootstrap phase variance threshold (radians)
    )";
    pyICU
       // Constructors
       .def(py::init([](const size_t buffer_lines, const size_t overlap_lines,
                        const bool use_phase_grad_neut, const bool use_intensity_neut,
                        const int phase_grad_win_size, const float neut_phase_grad_thr,
                        const float neut_intensity_thr, const float neut_correlation_thr,
                        const int trees_number, const int max_branch_len,
                        const float ratio_dxdy, const float init_corr_thr,
                        const float max_corr_thr, const float corr_incr_thr,
                        const float min_cc_area, const size_t num_bs_lines,
                        const size_t min_overlap_area, const float phase_var_thr)
                   {
                       ICU icu;
                       icu.numBufLines(buffer_lines);
                       icu.numOverlapLines(overlap_lines);
                       icu.usePhaseGradNeut(use_phase_grad_neut);
                       icu.useIntensityNeut(use_intensity_neut);
                       icu.phaseGradWinSize(phase_grad_win_size);
                       icu.neutPhaseGradThr(neut_phase_grad_thr);
                       icu.neutIntensityThr(neut_intensity_thr);
                       icu.neutCorrThr(neut_correlation_thr);
                       icu.numTrees(trees_number);
                       icu.maxBranchLen(max_branch_len);
                       icu.ratioDxDy(ratio_dxdy);
                       icu.initCorrThr(init_corr_thr);
                       icu.maxCorrThr(max_corr_thr);
                       icu.corrThrInc(corr_incr_thr);
                       icu.minCCAreaFrac(min_cc_area);
                       icu.numBsLines(num_bs_lines);
                       icu.minBsPts(min_overlap_area);
                       icu.bsPhaseVarThr(phase_var_thr);
                       return icu;
                   }),
                py::arg("buffer_lines")=3700,
                py::arg("overlap_lines")=200,
                py::arg("use_phase_grad_neut")=false,
                py::arg("use_intensity_neut")=false,
                py::arg("phase_grad_win_size")=5,
                py::arg("neut_phase_grad_thr")=3.0,
                py::arg("neut_intensity_thr")=8.0,
                py::arg("neut_correlation_thr")=0.8,
                py::arg("trees_number")=7,
                py::arg("max_branch_length")=64,
                py::arg("ratio_dxdy")= 1.0,
                py::arg("init_corr_thr")=0.1,
                py::arg("max_corr_thr")=0.9,
                py::arg("corr_incr_thr")=0.1,
                py::arg("min_cc_area")=0.003125,
                py::arg("num_bs_lines")=16,
                py::arg("min_overlap_area")=16,
                py::arg("phase_var_thr")=8.0
                )
       .def("unwrap", py::overload_cast<Raster&, Raster&, Raster&, Raster&, unsigned int>(&ICU::unwrap),
               py::arg("unw_igram"),
               py::arg("connected_components"),
               py::arg("igram"),
               py::arg("corr"),
               py::arg("seed")=0,
               py::call_guard<py::gil_scoped_release>(),
               R"(
       Perform phase unwrapping using the ICU algorithm
       
       Parameters
       ----------
       unw_igram: Raster
         Output unwrapped phase (radians)

       connected_components : Raster
         Connected components layer

       igram: Raster
         Input interferogram

       corr: Raster
         Interferometric correlation

       seed : int
         Seed value to initialize ICU
       )")
       
       // Properties
       .def_property("buffer_lines",
               py::overload_cast<>(&ICU::numBufLines, py::const_),
               py::overload_cast<size_t>(&ICU::numBufLines))
       .def_property("overlap_lines",
               py::overload_cast<>(&ICU::numOverlapLines, py::const_),
               py::overload_cast<size_t>(&ICU::numOverlapLines))
       .def_property("use_phase_grad_neut",
               py::overload_cast<>(&ICU::usePhaseGradNeut, py::const_),
               py::overload_cast<bool>(&ICU::usePhaseGradNeut))
       .def_property("use_intensity_neut",
               py::overload_cast<>(&ICU::useIntensityNeut, py::const_),
               py::overload_cast<bool>(&ICU::useIntensityNeut))
       .def_property("phase_grad_win_size",
               py::overload_cast<>(&ICU::phaseGradWinSize, py::const_),
               py::overload_cast<int>(&ICU::phaseGradWinSize))
       .def_property("neut_phase_grad_thr",
               py::overload_cast<>(&ICU::neutPhaseGradThr, py::const_),
               py::overload_cast<float>(&ICU::neutPhaseGradThr))
       .def_property("neut_intensity_thr",
               py::overload_cast<>(&ICU::neutIntensityThr, py::const_),
               py::overload_cast<float>(&ICU::neutIntensityThr))
       .def_property("neut_correlation_thr",
               py::overload_cast<>(&ICU::neutCorrThr, py::const_),
               py::overload_cast<float>(&ICU::neutCorrThr))
       .def_property("trees_number",
               py::overload_cast<>(&ICU::numTrees, py::const_),
               py::overload_cast<int>(&ICU::numTrees))
       .def_property("max_branch_length",
               py::overload_cast<>(&ICU::maxBranchLen, py::const_),
               py::overload_cast<int>(&ICU::maxBranchLen))
       .def_property("ratio_dxdy",
               py::overload_cast<>(&ICU::ratioDxDy, py::const_),
               py::overload_cast<float>(&ICU::ratioDxDy))
       .def_property("init_corr_thr",
               py::overload_cast<>(&ICU::initCorrThr, py::const_),
               py::overload_cast<float>(&ICU::initCorrThr))
       .def_property("max_corr_thr",
               py::overload_cast<>(&ICU::maxCorrThr, py::const_),
               py::overload_cast<float>(&ICU::maxCorrThr))
       .def_property("corr_incr_thr",
               py::overload_cast<>(&ICU::corrThrInc, py::const_),
               py::overload_cast<float>(&ICU::corrThrInc))
       .def_property("min_cc_area",
               py::overload_cast<>(&ICU::minCCAreaFrac, py::const_),
               py::overload_cast<float>(&ICU::minCCAreaFrac))
       .def_property("num_bs_lines",
               py::overload_cast<>(&ICU::numBsLines, py::const_),
               py::overload_cast<size_t>(&ICU::numBsLines))
       .def_property("min_overlap_area",
               py::overload_cast<>(&ICU::minBsPts, py::const_),
               py::overload_cast<size_t>(&ICU::minBsPts))
       .def_property("phase_var_thr",
               py::overload_cast<>(&ICU::bsPhaseVarThr, py::const_),
               py::overload_cast<float>(&ICU::bsPhaseVarThr))
       
       ;
}
