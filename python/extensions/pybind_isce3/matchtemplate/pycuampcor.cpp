#include "pycuampcor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <isce3/matchtemplate/pycuampcor/cuAmpcorController.h>
#include <isce3/matchtemplate/pycuampcor/cuAmpcorParameter.h>

void addbinding_pycuampcor_cpu(pybind11::module& m)
{
    using str = std::string;
    using cls = isce3::matchtemplate::pycuampcor::cuAmpcorController;

    pybind11::class_<cls>(m, "PyCPUAmpcor")
        .def(pybind11::init<>())

        // define a trivial binding for a controller method
#define DEF_METHOD(name) def(#name, &cls::name)

        // define a trivial getter/setter for a controller parameter
#define DEF_PARAM_RENAME(T, pyname, cppname) \
        def_property(#pyname, [](const cls& self) -> T { \
            return self.param->cppname; \
        }, [](cls& self, const T i) { \
            self.param->cppname = i; \
        })

        // same as above, for even more trivial cases where pyname == cppname
#define DEF_PARAM(T, name) DEF_PARAM_RENAME(T, name, name)

        .DEF_PARAM(int, algorithm)
        .DEF_PARAM(int, deviceID)
        .DEF_PARAM(int, nStreams)
        .DEF_PARAM(int, derampMethod)

        .DEF_PARAM(str, referenceImageName)
        .DEF_PARAM(int, referenceImageHeight)
        .DEF_PARAM(int, referenceImageWidth)
        .DEF_PARAM(str, secondaryImageName)
        .DEF_PARAM(int, secondaryImageHeight)
        .DEF_PARAM(int, secondaryImageWidth)

        .DEF_PARAM(int, numberWindowDown)
        .DEF_PARAM(int, numberWindowAcross)

        .DEF_PARAM_RENAME(int, windowSizeHeight, windowSizeHeightRaw)
        .DEF_PARAM_RENAME(int, windowSizeWidth,  windowSizeWidthRaw)

        .DEF_PARAM(str, offsetImageName)
        .DEF_PARAM(str, grossOffsetImageName)
        .DEF_PARAM(int, mergeGrossOffset)
        .DEF_PARAM(str, snrImageName)
        .DEF_PARAM(str, covImageName)
        .DEF_PARAM(str, corrImageName)

        .DEF_PARAM(int, rawDataOversamplingFactor)
        .DEF_PARAM(int, corrStatWindowSize)

        .DEF_PARAM(int, numberWindowDownInChunk)
        .DEF_PARAM(int, numberWindowAcrossInChunk)

        .DEF_PARAM(int, useMmap)

        .DEF_PARAM_RENAME(int, halfSearchRangeAcross, halfSearchRangeAcrossRaw)
        .DEF_PARAM_RENAME(int, halfSearchRangeDown,   halfSearchRangeDownRaw)

        .DEF_PARAM_RENAME(int, referenceStartPixelAcrossStatic, referenceStartPixelAcross0)
        .DEF_PARAM_RENAME(int, referenceStartPixelDownStatic,   referenceStartPixelDown0)

        .DEF_PARAM_RENAME(int, corrSurfaceOverSamplingMethod, oversamplingMethod)
        .DEF_PARAM_RENAME(int, corrSurfaceOverSamplingFactor, oversamplingFactor)

        .DEF_PARAM_RENAME(int, mmapSize, mmapSizeInGB)

        .DEF_PARAM_RENAME(int, skipSampleDown,   skipSampleDownRaw)
        .DEF_PARAM_RENAME(int, skipSampleAcross, skipSampleAcrossRaw)
        .DEF_PARAM_RENAME(int, corrSurfaceZoomInWindow, zoomWindowSize)

        .DEF_METHOD(runAmpcor)

        .def("checkPixelInImageRange", [](const cls& self) {
            self.param->checkPixelInImageRange();
        })

        .def("setupParams", [](cls& self) {
            self.param->setupParameters();
        })

        .def("setConstantGrossOffset", [](cls& self, const int goDown,
                                                     const int goAcross) {
            self.param->setStartPixels(
                    self.param->referenceStartPixelDown0,
                    self.param->referenceStartPixelAcross0,
                    goDown, goAcross);
        })
        .def("setVaryingGrossOffset", [](cls& self, std::vector<int> vD,
                                                    std::vector<int> vA) {
            self.param->setStartPixels(
                    self.param->referenceStartPixelDown0,
                    self.param->referenceStartPixelAcross0,
                    vD.data(), vA.data());
        })
        ;
}
