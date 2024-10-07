#include "Modulate.h"

#include <isce3/core/EMatrix.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Poly2d.h>
#include <isce3/image/Modulate.h>
#include <isce3/product/RadarGridParameters.h>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


template<typename AzRgFunc = isce3::core::Poly2d>
void addbindings_modulate(py::module & m)
{
    m.def(
        "_get_modulation_phase",
        py::overload_cast<
            isce3::image::modulate::ArrayRef2D<std::complex<float>>,
            const AzRgFunc&,
            const isce3::product::RadarGridParameters&,
            const bool,
            const std::complex<float>
        >(&isce3::image::modulate::getModulationPhase<AzRgFunc>),
        py::arg("out"),
        py::arg("carrier_phase"),
        py::arg("radar_grid"),
        py::arg("conjugate"),
        py::arg("fill_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        R"(
        Acquire the phase of the given carrier of a radar scene.

        Parameters
        ----------
        out: numpy.ndarray (complex64)
            The output phase array to modify. Anything in this array will be
            overwritten.
        carrier_phase: isce3.core.LUT2d or isce3.core.Poly2d
            Carrier phase, in radian, as a function of azimuth and range.
        radar_grid: isce3.product.RadarGridParameters
            Parameters for the given radar grid.
        conjugate: bool
            If True, get the conjugate of the phase.
        fill_value: complex
            The value to fill out-of-bounds pixels with. Out-of-bounds pixels are
            defined here as any pixels that cannot be evaluated by the carrier_phase
            function. Poly2d functions do not have out-of-bounds pixels, but LUT2d
            functions may. Defaults to NaN + j*NaN.
        )"
    );


    m.def(
        "_get_modulation_phase_at_coords",
        py::overload_cast<
            isce3::image::modulate::ArrayRef2D<std::complex<float>>,
            const AzRgFunc&,
            const isce3::product::RadarGridParameters&,
            const isce3::image::modulate::ConstArrayRef2D<double>,
            const isce3::image::modulate::ConstArrayRef2D<double>,
            const bool,
            const std::complex<float>
        >(&isce3::image::modulate::_getModulationPhaseAtCoords<AzRgFunc>),
        py::arg("out"),
        py::arg("carrier_phase"),
        py::arg("radar_grid"),
        py::arg("azimuth_indices"),
        py::arg("range_indices"),
        py::arg("conjugate"),
        py::arg("fill_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        R"(
        Acquire the phase of the given carrier at each given index of a radar scene.

        Parameters
        ----------
        out: numpy.ndarray (complex64)
            The output phase array to modify. Anything in this array will be
            overwritten.
        carrier_phase: isce3.core.LUT2d or isce3.core.Poly2d
            Carrier phase, in radian, as a function of azimuth and range.
        radar_grid: isce3.product.RadarGridParameters
            Parameters for the given radar grid.
        azimuth_indices: numpy.ndarray (float64)
            Azimuth index of each output coordinate pixel in the given radar coordinate
            system. Must be the same shape as phase_data_block.
        range_indices: numpy.ndarray (float64)
            Range index of each output coordinate pixel in the given radar coordinate
            system. Must be the same shape as phase_data_block.
        conjugate: bool
            If True, get the conjugate of the phase.
        fill_value: complex
            The value to fill out-of-bounds pixels with. Out-of-bounds pixels are
            defined here as any pixels that cannot be evaluated by the carrier_phase
            function. Poly2d functions do not have out-of-bounds pixels, but LUT2d
            functions may. Defaults to NaN + j*NaN.
        )"
    );


    m.def(
        "_modulate",
        py::overload_cast<
            isce3::image::modulate::ArrayRef2D<std::complex<float>>,
            const AzRgFunc&,
            const isce3::product::RadarGridParameters&,
            const bool,
            const std::complex<float>
        >(&isce3::image::modulate::modulate<AzRgFunc>),
        py::arg("slc_data_block"),
        py::arg("carrier_phase"),
        py::arg("radar_grid"),
        py::arg("conjugate"),
        py::arg("fill_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        R"(
        Evaluate and modulate or demodulate the phase carrier onto the given SLC data
        block.

        Parameters
        ----------
        slc_data_block: numpy.ndarray (complex64)
            The block of SLC data to modulate.
        carrier_phase: isce3.core.LUT2d or isce3.core.Poly2d
            Carrier phase, in radian, as a function of azimuth and range. This phase
            will be modulated to or demodulated from the image.
        radar_grid: isce3.product.RadarGridParameters
            Parameters for the given radar grid.
        conjugate: bool, optional
            If True, modulate the conjugate of the phase.
        fill_value: complex
            The value to fill out-of-bounds pixels with. Out-of-bounds pixels are
            defined here as any pixels that cannot be evaluated by the carrier_phase
            function. Poly2d functions do not have out-of-bounds pixels, but LUT2d
            functions may. Defaults to NaN + j*NaN.
        )"
    );


    m.def(
        "_modulate_at_coords",
        py::overload_cast<
            isce3::image::modulate::ArrayRef2D<std::complex<float>>,
            const AzRgFunc&,
            const isce3::product::RadarGridParameters&,
            const isce3::image::modulate::ConstArrayRef2D<double>,
            const isce3::image::modulate::ConstArrayRef2D<double>,
            const bool,
            const std::complex<float>
        >(&isce3::image::modulate::_modulateAtCoords<AzRgFunc>),
        py::arg("slc_data_block"),
        py::arg("carrier_phase"),
        py::arg("radar_grid"),
        py::arg("azimuth_indices"),
        py::arg("range_indices"),
        py::arg("conjugate"),
        py::arg("fill_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        R"(
        Evaluate and modulate or demodulate the phase carrier onto the given SLC data
        block at the given indices.

        Parameters
        ----------
        slc_data_block: numpy.ndarray (complex64)
            The output phase array to modulate.
        carrier_phase: isce3.core.LUT2d or isce3.core.Poly2d
            Carrier phase, in radian, as a function of azimuth and range. This phase
            will be modulated to or demodulated from the image.
        radar_grid: isce3.product.RadarGridParameters
            Parameters for the given radar grid.
        azimuth_indices: numpy.ndarray (float64)
            Azimuth index of each output coordinate pixel in the given radar coordinate
            system. Must be the same shape as phase_data_block.
        range_indices: numpy.ndarray (float64)
            Range index of each output coordinate pixel in the given radar coordinate
            system. Must be the same shape as phase_data_block.
        conjugate: bool, optional
            If True, modulate the conjugate of the phase.
        fill_value: complex
            The value to fill out-of-bounds pixels with. Out-of-bounds pixels are
            defined here as any pixels that cannot be evaluated by the carrier_phase
            function. Poly2d functions do not have out-of-bounds pixels, but LUT2d
            functions may. Defaults to NaN + j*NaN.
        )"
    );
}

template void addbindings_modulate<isce3::core::LUT2d<double>>(py::module & m);
template void addbindings_modulate<isce3::core::Poly2d>(py::module & m);
