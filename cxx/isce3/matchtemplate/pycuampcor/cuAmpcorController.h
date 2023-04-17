/**
 * @file  cuAmpcorController.h
 * @brief The controller for running cuAmcor
 *
 * cuAmpController is the main processor, also interface to python
 * It determines the total number of windows, the starting pixels for each window.
 * It then divides windows into chunks (batches), and creates cuAmpcorChunk instances
 *  to process each chunk.
 * A chunk includes multiple windows, to maximize the use of GPU cores.
 * Different cuAmpcorChunk processors use different cuda streams, to overlap
 *  the kernel execution with data copying.
 */

// code guard
#ifndef CU_AMPCOR_CONTROLLER_H
#define CU_AMPCOR_CONTROLLER_H

#include <memory>

// dependencies
#include "cuAmpcorParameter.h"

namespace isce3::matchtemplate::pycuampcor {

class cuAmpcorController {
public:
    std::unique_ptr<cuAmpcorParameter> param;  ///< the parameter set
    // constructor
    cuAmpcorController();
    // run interface
    void runAmpcor();
};
#endif

} // namespace
