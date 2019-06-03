#pragma once
#ifndef ISCE_CUDA_CORE_EVENT_H
#define ISCE_CUDA_CORE_EVENT_H

#include <memory>

namespace isce { namespace cuda { namespace core {

/** Thin RAII wrapper around cudaEvent_t */
class Event {
public:
    /** Create an event object on the current CUDA device. */
    Event();

    /** Return the underlying cudaEvent_t object. */
    cudaEvent_t get() const { return *_event; }

private:
    std::shared_ptr<cudaEvent_t> _event;
};

bool operator==(Event, Event);

bool operator!=(Event, Event);

/** Wait for an event to complete. */
void synchronize(Event);

/**
 * Query an event's status.
 *
 * Returns true if all work captured by the event has completed.
 */
bool query(Event);

}}}

#endif

