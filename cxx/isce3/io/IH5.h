#pragma once

#include <H5Cpp.h>
#include <complex>
#include <iostream>
#include <isce3/core/Constants.h>
#include <isce3/except/Error.h>
#include <regex>
#include <string>
#include <type_traits>
#include <typeindex>
#include <valarray>
#include <array>
#include <vector>

namespace isce3 {
namespace io {

// Dataset chunking size
const hsize_t chunkSizeX = 128;
const hsize_t chunkSizeY = 128;

// Specific isce data type for HDF5
// May be stored elsewhere eventually
typedef struct float16 {
} float16;
typedef struct n1bit {
} n1bit;
typedef struct n2bit {
} n2bit;

// Parameters containers for HDF5 searching capability
struct findMeta {
    std::vector<std::string> outList;
    std::string searchStr;
    std::string searchType;
    std::string basePath;
};

/** Our derived dataset structure that includes utility functions */
class IDataSet : public H5::DataSet {

public:
    IDataSet() : H5::DataSet() {};
    IDataSet(const H5::DataSet& dset) : H5::DataSet(dset) {};
    IDataSet(const hid_t id) : H5::DataSet(id) {};

    // Metadata query

    /** Get a list of attributes attached to dataset */
    std::vector<std::string> getAttrs();

    /** Get a HDF5 DataSpace object corresponding to dataset or given attribute
     */
    H5::DataSpace getDataSpace(const std::string& v = "");

    /** Get the number of dimension of dataset or given attribute */
    int getRank(const std::string& v = "");

    /** Get the total number of elements contained in dataset or given attribute
     */
    int getNumElements(const std::string& v = "");

    /** Get the size of each dimension of the dataset or given attribute */
    std::vector<int> getDimensions(const std::string& v = "");

    /** Get the H5 data type of the dataset or given attribute */
    std::string getTypeClassStr(const std::string& v = "");

    /** Get the storage chunk size of the dataset */
    std::vector<int> getChunkSize();

    /** Get the number of bit used to store each dataset element */
    int getNumBits(const std::string& v = "");

    /** Generate GDALDataset Representation */
    std::string toGDAL() const;

    // Dataset reading queries

    /** Reading scalar (non string) dataset or attributes */
    template<typename T> inline void read(T& v, const std::string& att = "");

    /** Reading scalar string dataset or attributes */
    void read(std::string& v, const std::string& att = "");

    /** Reading multi-dimensional attribute in raw pointer */
    template<typename T> inline void read(T* buf, const std::string& att);

    /** Reading multi-dimensional attribute in vector */
    template<typename T>
    inline void read(std::vector<T>& buf, const std::string& att);

    /** Reading multi-dimensional attribute in valarray */
    template<typename T>
    inline void read(std::valarray<T>& buf, const std::string& att);

    // Function to read a dataset from file to memory variable. The input
    // parameter is a raw point/vector/valarray that will store the (partial)
    // multi-dimensional dataset. Input vector/valarray is initialized by
    // caller, but resized (if needed) by the function. Raw pointer have to be
    // allocated by caller. Partial extraction of data is possible using
    // startIn, countIn, and strideIn. startIn: Array containing the position of
    // the first element to read in all
    //          dimension. SizeIn size must be equal to the number of dimension
    //          of the dataset. If nullptr, startIn values are 0.
    // countIn: Array containing the number of elements to read in all
    // dimension.
    //          countIn size must be equal to the number of dimension of the
    //          dataset.
    // strideIn:Array containing the stride between elements to read in all
    // dimension.
    //          strideIn size must be equal to the number of dimension of the
    //          dataset. If nullptr, stridIn values are 1.
    //
    // Examples:
    // - Dataset contains a 3-bands raster. Dimensions are (100,100,3).
    // To retrieve the full second band:
    // startIn=[0,0,1], countIn=[100,100,1], strideIn=nullptr or [1,1,1]
    // To retrieve the first band, but only every other elements in X direction:
    // startIn=[0,0,0], countIn=[50,100,1], strideIn=[2,1,1]

    /** Reading multi-dimensional dataset in raw pointer */
    template<typename T>
    inline void read(T* buf, const int* startIn = nullptr,
                     const int* countIn = nullptr,
                     const int* strideIn = nullptr);

    /** Reading multi-dimensional dataset in raw pointer with std:slice
     * subsetting */
    template<typename T>
    inline void read(T* buf, const std::vector<std::slice>* slicesIn);

    /** Reading multi-dimensional dataset in raw pointer with std:gslice
     * subsetting */
    template<typename T> inline void read(T* buf, const std::gslice* gsliceIn);

    /** Reading multi-dimensional dataset in std::vector */
    template<typename T> inline void read(std::vector<T>& buf);

    /** Reading multi-dimensional dataset in vector */
    template<typename T>
    inline void read(std::vector<T>& buf, const std::vector<int>* startIn,
                     const std::vector<int>* countIn,
                     const std::vector<int>* strideIn);

    /** Reading multi-dimensional dataset in vector with std:slice subsetting */
    template<typename T>
    inline void read(std::vector<T>& buf,
                     const std::vector<std::slice>* slicesIn);

    /** Reading multi-dimensional dataset in vector with std:gslice subsetting
     */
    template<typename T>
    inline void read(std::vector<T>& buf, const std::gslice* gsliceIn);

    /** Reading multi-dimensional dataset in valarray */
    template<typename T> inline void read(std::valarray<T>& buf);

    /** Reading multi-dimensional dataset in valarray */
    template<typename T>
    inline void read(std::valarray<T>& buf, const std::valarray<int>* startIn,
                     const std::valarray<int>* countIn,
                     const std::valarray<int>* strideIn);

    /** Reading multi-dimensional dataset in valarray with std:slice subsetting
     */
    template<typename T>
    inline void read(std::valarray<T>& buf,
                     const std::vector<std::slice>* slicesIn);

    /** Reading multi-dimensional dataset in valarray with std:slice subsetting
     */
    template<typename T>
    inline void read(std::valarray<T>& buf, const std::gslice* gsliceIn);

    // Dataset writing queries

    /** Writing std::vector data into a dataset */
    template<typename T> inline void write(const std::vector<T>& buf);

    /** Writing std::vector data into a multi-dimensional dataset using
     * std::array for subsetting */
    template<typename T, size_t S>
    inline void write(const std::vector<T>& buf,
                      const std::array<int, S>& startIn,
                      const std::array<int, S>& countIn,
                      const std::array<int, S>& strideIn);

    /** Writing std::vector data into a multi-dimensional dataset using
     * std::slice for subsetting */
    template<typename T>
    inline void write(const std::vector<T>& buf,
                      const std::vector<std::slice>* slicesIn);

    /** Writing std::vector data into a multi-dimensional dataset using
     * std::gslice for subsetting */
    template<typename T>
    inline void write(const std::vector<T>& buf, const std::gslice* gsliceIn);

    /** Writing std::valarray data into a dataset */
    template<typename T> inline void write(const std::valarray<T>& buf);

    /** Writing std::valarray data into a multi-dimensional dataset using
     * std::array for subsetting */
    template<typename T, size_t S>
    inline void write(const std::valarray<T>& buf,
                      const std::array<int, S>& startIn,
                      const std::array<int, S>& countIn,
                      const std::array<int, S>& strideIn);

    /** Writing std::valarray data into a multi-dimensional dataset using
     * std::slice for subsetting */
    template<typename T>
    inline void write(const std::valarray<T>& buf,
                      const std::vector<std::slice>* slicesIn);

    /** Writing std::valarray data into a multi-dimensional dataset using
     * std::gslice for subsetting */
    template<typename T>
    inline void write(const std::valarray<T>& buf, const std::gslice* gsliceIn);

    /** Writing a raw pointer buffer into a dataset */
    template<typename T> inline void write(const T* buf, const size_t sz);

    /** Writing a raw pointer into a multi-dimensional dataset using std::array
     * for subsetting */
    template<typename T, size_t S>
    inline void write(const T* buf, const std::array<int, S>& startIn,
                      const std::array<int, S>& countIn,
                      const std::array<int, S>& strideIn);

    /** Writing a buffer into a multi-dimensional dataset using std::slice for
     * subsetting */
    template<typename T>
    inline void write(const T* buf, const std::vector<std::slice>* slicesIn);

    /** Writing a buffer into a multi-dimensional dataset using std::gslice for
     * subsetting */
    template<typename T>
    inline void write(const T* buf, const std::gslice* gsliceIn);

    /** Creating and writing a scalar as an attribute */
    template<typename T>
    inline void createAttribute(const std::string& name, const T& data);

    /** Creating and writing a std::vector data as a 1D-array attribute */
    template<typename T>
    inline void createAttribute(const std::string& name,
                                const std::vector<T>& values);

    /** Creating and writing a std::valarray data as a 1D-array attribute */
    template<typename T>
    inline void createAttribute(const std::string& name,
                                const std::valarray<T>& values);

    /** Creating and writing a std::vector data with dimensions (std::array)  as
     * an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims,
                                const std::vector<T>& values);

    /** Creating and writing a std::valarray data with dimensions (std::array)
     * as an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims,
                                const std::valarray<T>& values);

    /** Creating and writing a raw pointer with dimensions (std::array) to data
     * as an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims, const T* buffer);

    /** Get DataSpace corresponding to slice defined by start, count and stride
     */
    H5::DataSpace getDataSpace(const int* startIn, const int* countIn,
                               const int* strideIn);

    /** Get DataSpace corresponding to slice defined by a vector of slices */
    H5::DataSpace getDataSpace(const std::vector<std::slice>* sliceIn);

    /** Get DataSpace corresponding to a gslice */
    H5::DataSpace getDataSpace(const std::gslice* gsliceIn);

    /** Get DataSpace with a GDAL RasterIO-like interface */
    H5::DataSpace getDataSpace(const size_t xidx, const size_t yidx,
                               const size_t iowidth, const size_t iolength,
                               const size_t band);

private:
    template<typename T> void read(T* buffer, const H5::DataSpace& dspace);

    template<typename T>
    void createAttribute(const std::string& name, const H5::DataType& datatype,
                         const H5::DataSpace& dataspace, const T* buffer);

    template<typename T>
    void write(const T* buf, const H5::DataSpace& filespace);
};

class IGroup : public H5::Group {

public:
    IGroup() : H5::Group() {};
    IGroup(const H5::Group& group) : H5::Group(group) {};
    IGroup(hid_t group) : H5::Group(group) {};

    std::vector<std::string> getAttrs();

    /** Search function for given name in the group */
    std::vector<std::string> find(const std::string name,
                                  const std::string start = ".",
                                  const std::string type = "BOTH",
                                  const std::string path = "FULL");

    /** Return the path of the group from the file root */
    std::string getPathname();

    /** Return the H5::DataSpace of the given attribute */
    H5::DataSpace getDataSpace(const std::string& name);

    /** Return the number of elements in the given attribute */
    int getNumElements(const std::string& name);

    /** Reading scalar attribute given by name */
    template<typename T> inline void read(T& v, const std::string& att);

    /** Reading scalar string attribute given by name  */
    void read(std::string& v, const std::string& att);

    /** Reading multi-dimensional attribute in raw pointer */
    template<typename T> inline void read(T* buf, const std::string& att);

    /** Reading multi-dimensional attribute in vector */
    template<typename T>
    inline void read(std::vector<T>& buf, const std::string& att);

    /** Reading multi-dimensional attribute in valarray */
    template<typename T>
    inline void read(std::valarray<T>& buf, const std::string& att);

    /** Open a given dataset */
    IDataSet openDataSet(const H5std_string& name);

    /** Open a given group */
    IGroup openGroup(const H5std_string& name);

    /** Create a group within this group */
    IGroup createGroup(const H5std_string& name);

    /** Create a string scalar dataset and simultaneously write the data*/
    IDataSet createDataSet(const std::string& name, const std::string& data);

    /** Create a (non-string) scalar dataset and simultaneously write the data*/
    template<typename T>
    inline IDataSet createDataSet(const std::string& name, const T& data);

    /** Create a dataset (1D) and write the data from a vector container*/
    template<typename T>
    inline IDataSet createDataSet(const std::string& name,
                                  const std::vector<T>& data);

    /** Create a dataset (1D) and write the data from a valarray container*/
    template<typename T>
    inline IDataSet createDataSet(const std::string& name,
                                  const std::valarray<T>& data);

    /** Create a dataset (1D) and write a buffer's data*/
    template<typename T>
    inline IDataSet createDataSet(const std::string& name, const T* buffer,
                                  const size_t size);

    /** Create a dataset (nD) and write the data from a vector container*/
    template<typename T, typename T2, size_t S>
    inline IDataSet createDataSet(const std::string& name,
                                  const std::vector<T>& data,
                                  const std::array<T2, S>& dims);

    /** Create a dataset (nD) and write the data from a valarray container*/
    template<typename T, typename T2, size_t S>
    inline IDataSet createDataSet(const std::string& name,
                                  const std::valarray<T>& data,
                                  const std::array<T2, S>& dims);

    /** Create a dataset (nD) and write the data from a buffer*/
    template<typename T, typename T2, size_t S>
    inline IDataSet createDataSet(const std::string& name, const T* data,
                                  const std::array<T2, S>& dims);

    /** Create a datatset with compression options*/
    template<typename T, typename T2, size_t S>
    IDataSet createDataSet(const std::string& name,
                           const std::array<T2, S>& dims, const int chunk = 0,
                           const int shuffle = 0, const int deflate = 0);

    /** Creating and writing a scalar as an attribute */
    template<typename T>
    inline void createAttribute(const std::string& name, const T& data);

    /** Creating and writing a std::vector data as a 1D-array attribute */
    template<typename T>
    inline void createAttribute(const std::string& name,
                                const std::vector<T>& values);

    /** Creating and writing a std::valarray data as a 1D-array attribute */
    template<typename T>
    inline void createAttribute(const std::string& name,
                                const std::valarray<T>& values);

    /** Creating and writing a std::vector data with dimensions (std::array) as
     * an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims,
                                const std::vector<T>& values);

    /** Creating and writing a std::valarray data with dimensions (std::array)
     * as an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims,
                                const std::valarray<T>& values);

    /** Creating and writing a raw pointer to data with dimensions (std::array)
     * as an attribute */
    template<typename T, typename T2, size_t S>
    inline void createAttribute(const std::string& name,
                                const std::array<T2, S>& dims, const T* buffer);

private:
    template<typename T>
    void createAttribute(const std::string& name, const H5::DataType& datatype,
                         const H5::DataSpace& dataspace, const T* buffer);
};

class IH5File : public H5::H5File {
public:
    IH5File() : H5::H5File() {};

    /** @param[in] name Name of the Hdf5 file to open.
     *  @param[in] mode File opening mode
     * The different mode available are:
     * - r: read only, file must exist (default)
     * - w: read/write, file must exist
     * - x: create file, overwrite if exist
     * - a: create file, fails if exist */
    IH5File(const H5std_string& name, const char mode = 'r');

    void openFile(const H5std_string& name);

    /** Open a given dataset */
    IDataSet openDataSet(const H5std_string& name);

    /** Open a given group */
    IGroup openGroup(const H5std_string& name);

    /** Searching for given name in file */
    std::vector<std::string> find(const std::string name,
                                  const std::string start = "/",
                                  const std::string type = "BOTH",
                                  const std::string path = "FULL");

    /** Get filename of HDF5 file. */
    inline std::string filename() const { return getFileName(); }

    /** Create a group */
    IGroup createGroup(const H5std_string& name);
};
} // namespace io
} // namespace isce3

// Get inline implementations for IH5
#define ISCE_IO_IH5_ICC
#include "IH5.icc"
#undef ISCE_IO_IH5_ICC
