#pragma once

#include <isce3/cuda/image/ResampSlc.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::cuda::image::ResampSlc>&);
