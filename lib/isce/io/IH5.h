
#ifndef __ISCE_IO_H5_H__
#define __ISCE_IO_H5_H__

#include <iostream>
#include <vector>
#include <valarray>
#include <string>
#include <complex>
#include <regex>
#include <type_traits>

#include "H5Cpp.h"

//#include <pyre/journal.h>

namespace isce {
  namespace io {
        
     class IDataSet: public H5::DataSet {
            
        public:
                
           // Constructors

           IDataSet(): H5::DataSet(){};
           IDataSet(H5::DataSet &dset): H5::DataSet(dset){};


           // Metadata query

           /** Get a list of attributes attached to dataset */
           std::vector<std::string> getAttrs(); 

           /** Get a H5 DataSpace object corresponding to dataset or given attribute */
           H5::DataSpace getDataSpace(const std::string &v="");

           /** Get the number of dimension of dataset or given attribute */
           int getRank(const std::string &v = "");

           /** Get the total number of elements contained in dataset or given attribute */
           int getNumElements(const std::string &v = "");

           /** Get the size of each dimension of the dataset or given attribute */
           std::vector<int> getDimensions(const std::string &v = "");

           /** Get the H5 data type of the dataset or given attribute */
           std::string getTypeClassStr(const std::string &v = "");

           /** Get the storage chunk size of the dataset */
           std::vector<int> getChunkSize();

           /** Get the number of bit used to store each dataset element */
           int getNumBits(const std::string &v ="");



           // Reading queries

           /** Reading scalar (non string) dataset or attributes */
           template<typename T>
           void read(T &v, const std::string &att=""); 

           /** Reading scalar string dataset or attributes */
           void read(std::string &v, const std::string &att=""); 



           /** Reading multi-dimensional attribute in raw pointer */
           template<typename T>
           void read(T *buf, const std::string &att); 

           /** Reading multi-dimensional attribute in vector */
           template<typename T>
           void read(std::vector<T> &buf, const std::string &att); 

           /** Reading multi-dimensional attribute in valarray */
           template<typename T>
           void read(std::valarray<T> &buf, const std::string &att); 
      



           // Function to read a dataset from file to memory variable. The input parameter 
           // is a raw point/vector/valarray that will store the (partial) multi-dimensional 
           // dataset.
           // Input vector/valarray is initialized by caller, but resized (if needed) by the 
           // function. Raw pointer have to be allocated by caller. 
           // Partial extraction of data is possible using startIn, countIn, and strideIn.
           // startIn: Array containing the position of the first element to read in all
           //          dimension. SizeIn size must be equal to the number of dimension 
           //          of the dataset. If nullptr, startIn values are 0.
           // countIn: Array containing the number of elements to read in all dimension. 
           //          countIn size must be equal to the number of dimension of the dataset.
           // strideIn:Array containing the stride between elements to read in all dimension.
           //          strideIn size must be equal to the number of dimension of the dataset. 
           //          If nullptr, stridIn values are 1.
           //
           // Examples:
           // - Dataset contains a 3-bands raster. Dimensions are (100,100,3).
           // To retrieve the full second band: 
           // startIn=[0,0,1], countIn=[100,100,1], strideIn=nullptr or [1,1,1]
           // To retrieve the first band, but only every other elements in X direction:
           // startIn=[0,0,0], countIn=[50,100,1], strideIn=[2,1,1]



           /** Reading multi-dimensional dataset in raw pointer */
           template<typename T>
           void read(T *buf, const int * startIn = nullptr, 
		                     const int * countIn = nullptr, 
                             const int * strideIn = nullptr);
     
           
           /** Reading multi-dimensional dataset in vector */
           template<typename T>
           void read(std::vector<T> &buf);

           /** Reading multi-dimensional dataset in vector */
           template<typename T>
           void read(std::vector<T> &buf, const std::vector<int>  * startIn, 
		                                  const std::vector<int>  * countIn, 
                                          const std::vector<int>  * strideIn);


           /** Reading multi-dimensional dataset in valarray */
           template<typename T>
           void read(std::valarray<T> &buf);

           /** Reading multi-dimensional dataset in valarray */
           template<typename T>
           void read(std::valarray<T> &buf, const std::valarray<int> * startIn, 
                                            const std::valarray<int> * countIn, 
                                            const std::valarray<int> * strideIn);


        private:

           H5::DataSpace getReadDataSpace(const int * startIn, 
                                          const int * countIn,
                                          const int * strideIn);

           H5::DataSpace getReadMemorySpace(hsize_t nbElements);
     };





     class IH5File: public H5::H5File {
   
         public:

           // Constructors
           IH5File(const H5std_string &name) : 
               H5::H5File(name, H5F_ACC_RDONLY),
               _name(name) {};
   
           void openFile(const H5std_string &name);

           /** Open a given dataset */ 
           IDataSet openDataSet(const H5std_string &name);

           /** Searching for given name in file */
           std::vector<std::string> find(const std::string name, 
                                         const std::string start = "/", 
                                         const std::string type = "BOTH");

           /** Get filename of HDF5 file */
           inline std::string filename() const { return _name; }

         private:
           // The filename of the HDF5 file
           std::string _name;
     };


  }
}

#endif
