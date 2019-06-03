#include <isce/cuda/except/Error.h>

#include "Stream.h"

namespace isce { namespace cuda { namespace core {

Stream::Stream()
{
    // construct shared pointer with custom deleter to
    // clean up the stream object
    _stream = std::shared_ptr<cudaStream_t> {
        new cudaStream_t,
        [](cudaStream_t * stream) noexcept {
            cudaStreamDestroy(*stream);
            delete stream;
        }};

    // init stream
    checkCudaErrors( cudaStreamCreate(_stream.get()) );
}

Stream::Stream(std::nullptr_t)
{
    _stream = std::make_shared<cudaStream_t>();
}

Stream::operator bool() const
{
    return get();
}

void Stream::record_event(Event event) const
{
    checkCudaErrors( cudaEventRecord(event.get(), get()) );
}

Event Stream::record_event() const
{
    Event event;
    record_event(event);
    return event;
}

void Stream::wait_event(Event event) const
{
    checkCudaErrors( cudaStreamWaitEvent(get(), event.get(), 0) );
}

bool operator==(Stream lhs, Stream rhs)
{
    return lhs.get() == rhs.get();
}

bool operator!=(Stream lhs, Stream rhs)
{
    return !(lhs == rhs);
}

void synchronize(Stream stream)
{
    checkCudaErrors( cudaStreamSynchronize(stream.get()) );
}

bool query(Stream stream)
{
    cudaError_t status = cudaStreamQuery(stream.get());
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

