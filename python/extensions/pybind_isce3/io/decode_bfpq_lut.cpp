#include "decode_bfpq_lut.h"
#include <cstdint>
#include <complex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <stdexcept>

struct complex_u16 {
    std::uint16_t r, i;
};

void addbinding_decode_bfpq_lut(py::module &m)
{
    // macro allows use of structured type in array_t
    PYBIND11_NUMPY_DTYPE(complex_u16, r, i);
    m.def("decode_bfpq_lut", [](
        py::array_t<float> table,
        py::array_t<complex_u16> indices) {
            if (table.ndim() != 1) {
                throw std::runtime_error("Expected a 1D BFPQ look up table.");
            }
            if (indices.ndim() != 2) {
                throw std::runtime_error(
                    "Optimized decoder only works for 2D slices.");
            }
            // get dimensions
            const auto m = indices.shape(0), n = indices.shape(1);
            // allocate output
            py::array_t<std::complex<float>> out({m, n});
            // get views that don't check bounds on every operator()
            auto indices_ = indices.unchecked<2>();
            auto table_ = table.unchecked<1>();
            auto out_ = out.mutable_unchecked<2>();
            // run in parallel!
            #pragma omp parallel for collapse(2)
            for (py::ssize_t i = 0; i < m; ++i) {
                for (py::ssize_t j = 0; j < n; ++j) {
                    const auto index = indices_(i, j);
                    out_(i, j) = std::complex<float>(
                        table_(index.r),
                        table_(index.i)
                    );
                }
            }
            return out;
        },
        py::arg("table"), py::arg("indices"),
        R"(
        Look up complex valued indexes in BFPQ decode table,
        equivalent to `table[indices['r']] + 1j * table[indices['i']]`

        This is a highly specialized function provided to parallelize
        a particular task related to decoding NISAR L0B data.  Note that it
        only works for indices.ndim==2.

        Parameters
        ----------
        table : numpy.ndarray
            BFPQ look up table that provides the floating-point values
            associated with the composite key `(exponent << M) | mantissa`
            where M is the number of bits in the mantissa (including sign).
        indices : numpy.ndarray
            Array containing the data signal data, where the real and imaginary
            parts are each encoded as 16-bit unsigned integer indexes into
            `table`.

        Returns
        -------
        out : numpy.ndarray
            Complex floating-point data decoded from `indices`.
        )");
}
