#include <gtest/gtest.h>
#include <isce/cuda/core/Event.h>
#include <isce/cuda/core/Stream.h>
#include <isce/cuda/except/Error.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

using namespace isce::cuda::core;

using thrust::device_vector;
using thrust::host_vector;
using thrust::system::cuda::experimental::pinned_allocator;

template<typename T>
using pinned_host_vector = host_vector<T, pinned_allocator<T>>;

TEST(EventTest, Comparison)
{
    Event event1;
    Event event2 = event1;
    Event event3;

    EXPECT_TRUE( event1 == event2 );
    EXPECT_TRUE( event1 != event3 );
}

// spin for *count* cycles or *maxiters* iterations, whichever
// termination condition is first encountered
__global__
void spin(clock_t count, int maxiters = 100000)
{
    clock_t start = clock();
    clock_t offset = 0;

    for (int i = 0; i < maxiters; ++i) {
        offset = clock() - start;
        if (offset >= count) { break; }
    }
}

// cause the stream to busy-wait for approx *ms* milliseconds
void busy_wait(Stream stream, int ms)
{
    cudaDeviceProp prop;
    checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
    clock_t cycles = ms * prop.clockRate;
    spin<<<1, 1, 0, stream.get()>>>(cycles);
    checkCudaErrors( cudaPeekAtLastError() );
}

TEST(EventTest, Synchronize)
{
    // false on the host, true on the device
    pinned_host_vector<bool> h (1, false);
    device_vector<bool> d (1, true);

    Stream stream;
    Event event;

    // no work has been captured by the event yet
    EXPECT_TRUE( query(event) );

    // wait for ~200 ms then copy the value from the device to
    // the host asynchronously
    busy_wait(stream, 200);
    checkCudaErrors( cudaMemcpyAsync(h.data(), d.data().get(),
            sizeof(bool), cudaMemcpyDeviceToHost, stream.get()) );

    // record the work enqueued in the stream
    stream.record_event(event);
    EXPECT_FALSE( query(event) );

    // host thread does not wait for streamed operations above
    // to complete
    EXPECT_FALSE( h[0] );

    // wait for all work captured to complete
    synchronize(event);

    EXPECT_TRUE( h[0] );
    EXPECT_TRUE( query(event) );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

