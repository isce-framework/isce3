#pragma once
#ifndef ISCE_CUDA_CORE_STREAM_H
#define ISCE_CUDA_CORE_STREAM_H

#include <memory>

#include "Event.h"

namespace isce { namespace cuda { namespace core {

/** Thin RAII wrapper around cudaStream_t */
class Stream {
public:
    /** Create an asynchronous stream object on the current CUDA device. */
    Stream();

    /** Create a NULL stream object (which causes implicit synchronization). */
    Stream(std::nullptr_t);

    /** Return true if the Stream is not NULL. */
    explicit operator bool() const;

    /** Return the underlying cudaStream_t object. */
    cudaStream_t get() const { return *_stream; }

    /** Record an event to capture the contents of the stream. */
    void record_event(Event) const;

    /** Record an event to capture the contents of the stream. */
    Event record_event() const;

    /** Wait for an event to complete. */
    void wait_event(Event) const;

private:
    std::shared_ptr<cudaStream_t> _stream;
};

bool operator==(Stream, Stream);

bool operator!=(Stream, Stream);

/** Wait for all work enqueued by the stream to complete. */
void synchronize(Stream);

/**
 * Query a stream's status.
 *
 * Returns true if all work enqueued by the stream has completed.
 */
bool query(Stream);

}}}

#endif

