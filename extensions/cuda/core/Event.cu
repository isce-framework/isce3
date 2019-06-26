#include <isce/cuda/except/Error.h>

#include "Event.h"

namespace isce { namespace cuda { namespace core {

Event::Event()
{
    // construct shared pointer with custom deleter to clean up the event object
    _event = std::shared_ptr<cudaEvent_t> {
        new cudaEvent_t,
        [](cudaEvent_t * event) noexcept {
            cudaEventDestroy(*event);
            delete event;
        }};

    // init event with flags for best runtime performance
    checkCudaErrors( cudaEventCreateWithFlags(_event.get(),
            cudaEventDisableTiming | cudaEventBlockingSync) );
}

bool operator==(Event lhs, Event rhs)
{
    return lhs.get() == rhs.get();
}

bool operator!=(Event lhs, Event rhs)
{
    return !(lhs == rhs);
}

void synchronize(Event event)
{
    checkCudaErrors( cudaEventSynchronize(event.get()) );
}

bool query(Event event)
{
    cudaError_t status = cudaEventQuery(event.get());
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        return false;
    }

    // this line should always throw - return statement
    // is just needed to prevent compiler warnings
    checkCudaErrors(status);
    return false;
}

}}}

