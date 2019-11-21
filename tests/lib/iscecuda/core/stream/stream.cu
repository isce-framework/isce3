#include <gtest/gtest.h>
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

TEST(StreamTest, Constructor)
{
    {
        Stream stream;
        EXPECT_NE( stream.get(), nullptr );
    }

    {
        Stream stream = 0;
        EXPECT_EQ( stream.get(), nullptr );
    }
}

TEST(StreamTest, OperatorBool)
{
    {
        Stream stream;
        EXPECT_TRUE( stream );
    }

    {
        Stream stream = 0;
        EXPECT_FALSE( stream );
    }
}

TEST(StreamTest, Comparison)
{
    Stream stream1;
    Stream stream2 = stream1;
    Stream stream3;

    EXPECT_TRUE( stream1 == stream2 );
    EXPECT_TRUE( stream1 != stream3 );
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

TEST(StreamTest, Synchronize)
{
    // false on the host, true on the device
    pinned_host_vector<bool> h (1, false);
    device_vector<bool> d (1, true);

    Stream stream;

    // no work has been added to the stream yet
    EXPECT_TRUE( query(stream) );

    // wait for ~200 ms then copy the value from the device to
    // the host asynchronously
    busy_wait(stream, 200);
    checkCudaErrors( cudaMemcpyAsync(h.data(), d.data().get(),
            sizeof(bool), cudaMemcpyDeviceToHost, stream.get()) );

    // host thread does not wait for streamed operations above
    // to complete
    EXPECT_FALSE( h[0] );
    EXPECT_FALSE( query(stream) );

    // wait for all streamed work to complete
    synchronize(stream);

    EXPECT_TRUE( h[0] );
    EXPECT_TRUE( query(stream) );
}

TEST(StreamTest, WaitEvent)
{
    // false on the host, true on the device
    pinned_host_vector<bool> h (1, false);
    device_vector<bool> d (1, true);

    Stream stream1;
    Stream stream2;

    // first stream waits for ~200 ms then copies the value
    // from the device to the host asynchronously
    busy_wait(stream1, 200);
    checkCudaErrors( cudaMemcpyAsync(h.data(), d.data().get(),
            sizeof(bool), cudaMemcpyDeviceToHost, stream1.get()) );

    // record the work enqueued in the stream
    Event event = stream1.record_event();
    EXPECT_FALSE( query(event) );

    // second stream is asynchronous w.r.t. first stream
    synchronize(stream2);
    EXPECT_FALSE( h[0] );

    // make second stream wait for the recorded operations to complete
    // then make host thread wait for second stream
    stream2.wait_event(event);
    synchronize(stream2);

    EXPECT_TRUE( h[0] );
    EXPECT_TRUE( query(event) );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

