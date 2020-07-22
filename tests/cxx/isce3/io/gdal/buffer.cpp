#include <array>
#include <gtest/gtest.h>
#include <vector>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/Buffer.h>

using isce3::io::gdal::Buffer;
using isce3::io::gdal::TypedBuffer;

TEST(BufferTest, Buffer)
{
    using T = float;
    GDALDataType datatype = GDT_Float32;

    int length = 4;
    int width = 5;
    std::size_t rowstride = width * sizeof(T);
    std::size_t colstride = sizeof(T);

    std::vector<T> v(length * width);
    Buffer buffer(static_cast<void *>(v.data()), datatype, {length, width}, {rowstride, colstride});

    EXPECT_EQ( buffer.data(), static_cast<void *>(v.data()) );
    EXPECT_EQ( buffer.datatype(), datatype );
    EXPECT_EQ( buffer.itemsize(), sizeof(T) );
    EXPECT_EQ( buffer.length(), length );
    EXPECT_EQ( buffer.width(), width );
    EXPECT_EQ( buffer.rowstride(), rowstride );
    EXPECT_EQ( buffer.colstride(), colstride );
    EXPECT_EQ( buffer.access(), GA_Update );

    std::array<int, 2> shape = { length, width };
    std::array<std::size_t, 2> strides = { rowstride, colstride };
    EXPECT_EQ( buffer.shape(), shape );
    EXPECT_EQ( buffer.strides(), strides );
}

TEST(TypedBufferTest, TypedBuffer)
{
    using T = float;
    GDALDataType datatype = GDT_Float32;

    int length = 4;
    int width = 5;
    std::size_t rowstride = width * sizeof(T);
    std::size_t colstride = sizeof(T);

    std::vector<T> v(length * width);
    TypedBuffer<T> buffer(v.data(), {length, width}, {rowstride, colstride});

    EXPECT_EQ( buffer.data(), v.data() );
    EXPECT_EQ( buffer.datatype(), datatype );
    EXPECT_EQ( buffer.itemsize(), sizeof(T) );
    EXPECT_EQ( buffer.length(), length );
    EXPECT_EQ( buffer.width(), width );
    EXPECT_EQ( buffer.rowstride(), rowstride );
    EXPECT_EQ( buffer.colstride(), colstride );
    EXPECT_EQ( buffer.access(), GA_Update );

    std::array<int, 2> shape = { length, width };
    std::array<std::size_t, 2> strides = { rowstride, colstride };
    EXPECT_EQ( buffer.shape(), shape );
    EXPECT_EQ( buffer.strides(), strides );
}

TEST(BufferTest, Cast)
{
    using T = float;
    GDALDataType datatype = GDT_Float32;

    int length = 4;
    int width = 5;
    std::size_t rowstride = width * sizeof(T);
    std::size_t colstride = sizeof(T);

    std::vector<T> v(length * width);
    Buffer buffer(static_cast<void *>(v.data()), datatype, {length, width}, {rowstride, colstride});

    TypedBuffer<T> typed_buffer = buffer.cast<T>();

    EXPECT_EQ( typed_buffer.data(), static_cast<T *>(buffer.data()) );
    EXPECT_EQ( typed_buffer.datatype(), buffer.datatype() );
    EXPECT_EQ( typed_buffer.itemsize(), buffer.itemsize() );
    EXPECT_EQ( typed_buffer.shape(), buffer.shape() );
    EXPECT_EQ( typed_buffer.length(), buffer.length() );
    EXPECT_EQ( typed_buffer.width(), buffer.width() );
    EXPECT_EQ( typed_buffer.strides(), buffer.strides() );
    EXPECT_EQ( typed_buffer.rowstride(), buffer.rowstride() );
    EXPECT_EQ( typed_buffer.colstride(), buffer.colstride() );
    EXPECT_EQ( typed_buffer.access(), buffer.access() );
}

TEST(BufferTest, CastInvalidType)
{
    using T = float;
    GDALDataType datatype = GDT_Float32;

    int length = 4;
    int width = 5;
    std::size_t rowstride = width * sizeof(T);
    std::size_t colstride = sizeof(T);

    std::vector<T> v(length * width);
    Buffer buffer(static_cast<void *>(v.data()), datatype, {length, width}, {rowstride, colstride});

    using U = double;
    EXPECT_THROW( { buffer.cast<U>(); }, isce3::except::RuntimeError );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
