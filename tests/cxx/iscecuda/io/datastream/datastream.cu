#include <gtest/gtest.h>
#include <isce/cuda/io/DataStream.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

TEST(DataStreamTest, FileDataStream)
{
    std::string filename = "./tmpfile";
    isce::cuda::core::Stream stream;
    std::size_t buffer_size = 16;

    isce::cuda::io::FileDataStream datastream (filename, stream, buffer_size);

    EXPECT_EQ( datastream.filename(), filename );
    EXPECT_EQ( datastream.stream(), stream );
    EXPECT_EQ( datastream.buffer_size(), buffer_size );
}

TEST(DataStreamTest, SetStream)
{
    std::string filename = "./tmpfile";
    isce::cuda::core::Stream stream1;
    std::size_t buffer_size = 16;

    isce::cuda::io::FileDataStream datastream (filename, stream1, buffer_size);

    isce::cuda::core::Stream stream2;
    datastream.set_stream(stream2);

    EXPECT_EQ( datastream.stream(), stream2 );
}

TEST(DataStreamTest, ResizeBuffer)
{
    std::string filename = "./tmpfile";
    isce::cuda::core::Stream stream;
    std::size_t buffer_size = 16;

    isce::cuda::io::FileDataStream datastream (filename, stream, buffer_size);

    std::size_t new_buffer_size = 32;
    datastream.resize_buffer(new_buffer_size);

    EXPECT_EQ( datastream.buffer_size(), new_buffer_size );
}

TEST(DataStreamTest, FileStoreThenLoad)
{
    // fill input device vector with {0, 1, 2, ..., 15}
    std::size_t size = 16;
    thrust::device_vector<int> src (size);
    thrust::sequence(src.begin(), src.end());

    // output device vector
    thrust::device_vector<int> dst (size);

    std::string filename = "./tmpfile";
    isce::cuda::core::Stream stream;

    isce::cuda::io::FileDataStream datastream (filename, stream);

    // write data to file
    std::size_t count = size * sizeof(int);
    datastream.store(src.data().get(), 0, count);

    // load data from file
    datastream.load(dst.data().get(), 0, count);

    // wait for asynchronous operations to complete
    isce::cuda::core::synchronize(stream);
    EXPECT_EQ( src, dst );
}

TEST(DataStreamTest, RasterDataStream)
{
    std::size_t width = 4;
    std::size_t length = 4;
    isce::io::Raster raster ("./tmpraster", width, length, 1, GDT_Int32, "ENVI");

    isce::cuda::core::Stream stream;
    std::size_t buffer_size = width * length;

    isce::cuda::io::RasterDataStream datastream (&raster, stream, buffer_size);

    EXPECT_EQ( datastream.raster(), &raster );
    EXPECT_EQ( datastream.stream(), stream );
    EXPECT_EQ( datastream.buffer_size(), buffer_size );
}

TEST(DataStreamTest, RasterStoreThenLoad)
{
    // fill input device vector with {0, 1, 2, ..., 15}
    std::size_t width = 4;
    std::size_t length = 4;
    thrust::device_vector<int> src (width * length);
    thrust::sequence(src.begin(), src.end());

    // output device vector
    thrust::device_vector<int> dst (width * length);

    isce::io::Raster raster ("./tmpraster", width, length, 1, GDT_Int32, "ENVI");
    isce::cuda::core::Stream stream;

    isce::cuda::io::RasterDataStream datastream (&raster, stream);

    // write data to file
    datastream.store(src.data().get(), 0, 0, width, length);

    // load data from file
    datastream.load(dst.data().get(), 0, 0, width, length);

    // wait for asynchronous operations to complete
    isce::cuda::core::synchronize(stream);
    EXPECT_EQ( src, dst );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

