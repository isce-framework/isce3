
#include "../core/Constants.h"

#include "IH5.h"


template<class T>
H5::DataType memType2 (const std::type_info &ti, T &d) {

//    H5T_class_t type = d.getTypeClass();

	if (ti == typeid(char))
		return H5::PredType::NATIVE_CHAR;
	else if (ti == typeid(unsigned char))
        return H5::PredType::NATIVE_UCHAR;
	else if (ti == typeid(signed char))
		return H5::PredType::NATIVE_SCHAR;
	else if (ti == typeid(short))
		return H5::PredType::NATIVE_SHORT;
	else if (ti == typeid(unsigned short))
		return H5::PredType::NATIVE_USHORT;
	else if (ti == typeid(int))
		return H5::PredType::NATIVE_INT;
	else if (ti == typeid(unsigned int))
		return H5::PredType::NATIVE_UINT;
	else if (ti == typeid(long))
		return H5::PredType::NATIVE_LONG;
	else if (ti == typeid(unsigned long))
		return H5::PredType::NATIVE_ULONG;
	else if (ti == typeid(long long))
		return H5::PredType::NATIVE_LLONG;
	else if (ti == typeid(unsigned long long))
		return H5::PredType::NATIVE_ULLONG;
	else if (ti == typeid(float))
		return H5::PredType::NATIVE_FLOAT;
	else if (ti == typeid(double))
		return H5::PredType::NATIVE_DOUBLE;
	else if (ti == typeid(long double))
		return H5::PredType::NATIVE_LDOUBLE;
	else if (ti == typeid(std::complex<float>))
		return d.getCompType();
	else if (ti == typeid(std::complex<double>))
		return d.getCompType();
	else if (ti == typeid(std::string)) {
		return d.getStrType();
    }
    else if (ti == typeid(isce::core::FixedString)) {
        H5::StrType strdatatype(H5::PredType::C_S1, 50);
        return strdatatype;
    } else 
	    return H5::PredType::NATIVE_FLOAT;//TODO error instead
}


template <class T1, class T2>
H5::DataType memType (std::valarray<T1> &v, T2 &d) {
     const std::type_info &ti = typeid(T1);
     return memType2(ti, d);
}

template <class T1, class T2>
H5::DataType memType (std::vector<T1> &v, T2 &d) {
     const std::type_info &ti = typeid(T1);
     return memType2(ti, d);
}

template <class T1, class T2>
H5::DataType memType (T1 &v, T2 &d) {
     const std::type_info &ti = typeid(T1);
     return memType2(ti, d);
}



void attrsNames(H5::H5Object &loc, H5std_string nameAttr, void *opdata){
   auto up = reinterpret_cast<std::vector<std::string> *>(opdata);
   up->push_back(nameAttr);
}



// User function to be passed to HDF5 *visit* function that recursively return 
// all objects contained in the file (or from a given path) and to which is 
// applied the user function.
// Implementation note: The function signature allows one user parameter (opdata)
// to be provided. We need 3 parameters: the vector containing the found objects, 
// the searched string, whether the user wants only DATASET, GROUP or BOTH, and if
// the user wants the search to occur from a given start in the tree.
// To keep things simple, the search string and object needed are stored in the 
// vector in the first and second location respectively. They are removed from 
// the vector once the search is completed  
herr_t matchName(hid_t loc_id, const char *name, const H5O_info_t *info, 
                     void *opdata) {
   
   auto outList = reinterpret_cast<std::vector<std::string> *>(opdata);

   // Check if the current object has a name containing the searched string.
   // If it does, check that its type (group/dataset) matches the user need.
   // Keep name of object (i.e., path+name) if success
   
   std::regex pattern((*outList)[0]);

   if (std::regex_search(name, pattern)) {
      if(info->type == H5O_TYPE_GROUP && (*outList)[1].compare("GROUP") == 0)
      outList->push_back((*outList)[2]+name);
      else if (info->type == H5O_TYPE_DATASET && (*outList)[1].compare("DATASET") == 0)
      outList->push_back((*outList)[2]+name);
      else if ((*outList)[1].compare("BOTH") == 0)
           outList->push_back((*outList)[2]+name);
      else {}
   }

   return 0;

}






H5::DataSpace isce::io::IDataSet::getDataSpace(const std::string &v) {

    H5::DataSpace dspace;

    // Looking for the dataset itself
    if (v.empty()) 
        // Open the dataspace of the current dataset
        dspace = H5::DataSet::getSpace();
    // Looking for the attribute of name v contained in the dataset itself
    else if (attrExists(v)){    
       // Open the attribute of the given name
       H5::Attribute attr = openAttribute(v);

       // Open the dataspace of the current attribute
       dspace = attr.getSpace();
        
       // Close the attribute
       attr.close();
    }
    else {}//TODO throw exception

    return dspace;
}



// Get the number of dimension of the current dataset
// scalar:0, 1D array:1, 2D array:2, ...
int isce::io::IDataSet::getRank(const std::string &v) {

    // Get the dataspace of dataset or attribute
    H5::DataSpace dspace = getDataSpace(v);

    // Get the rank of the data
    int rank = dspace.getSimpleExtentNdims();

    // Close the dataspace
    dspace.close();

    return rank;    
};



// Get the total number of elements in the current
// dataset
int isce::io::IDataSet::getNumElements(const std::string &v) {

    // Get the dataspace of dataset or attribute
    H5::DataSpace dspace = getDataSpace(v);
  
    // Get the total number of elements in the dataset
    int nbElements = dspace.getSimpleExtentNpoints();

    // Close the dataspace
    dspace.close();

    return nbElements;
}



// Get the dimension of the current dataset. Dimension
// are returned as a vector. A scalar dataset will 
// return an empty vector. 
std::vector<int> isce::io::IDataSet::getDimensions(const std::string &v) {

    // Get the dataspace of dataset or attribute
    H5::DataSpace dspace = getDataSpace(v);

    // Get the rank of the dataset, i.e., the number of 
    // dimension
    int rank = dspace.getSimpleExtentNdims();

    // Initialize the vector that will be returned to the
    // caller
    std::vector<int> outDims;

    // If dataset is not a scalar proceed
    if (rank) {
	   // Get dimensions    
       hsize_t dims[rank];
       dspace.getSimpleExtentDims(dims, NULL);

       // Fill the output vector
       for (int i=0; i<rank; i++)
          outDims.push_back(dims[i]);
    }

    // Close the dataspace
    dspace.close();

    return outDims;
};


// Get the HDF5 type of the data contained in the current dataset. This 
// information is useful to provide the read function with a variable/vector of
// the proper type. The HFD5 library will do type conversion when possible, but 
// will throw an error if types are incompatible.
std::string isce::io::IDataSet::getTypeClassStr(const std::string &v) {

    H5T_class_t type;

    // Looking for the dataset data type
    if (v.empty())
        // Get the data type class of the current dataset
        type = H5::DataSet::getTypeClass();  
    // Looking for the attribute v data type
    else if (attrExists(v)){
        // Open the attribute of the given name
        H5::Attribute attr = H5::DataSet::openAttribute(v);
    
        // Get the Class type of the attribute type
        type = attr.getTypeClass();  

        // Close the open attribute
        attr.close();
    }
    else {
       return "Unknown data type";
       //TODO Should throw an error
    }

    // Return a human-readable attribute type
    switch (type) {
        case H5T_STRING:
            return "H5T_STRING";
            break;
        case H5T_INTEGER:
            return "H5T_INTEGER";
            break;
        case H5T_FLOAT:
            return "H5T_FLOAT";
            break;
        case H5T_TIME:
            return "H5T_TIME";
            break;
        case H5T_BITFIELD:
            return "H5T_BITFIELD";
            break;
        case H5T_OPAQUE:
            return "H5T_OPAQUE";
            break;
        case H5T_COMPOUND:
            return "H5T_COMPOUND";
            break;
        case H5T_REFERENCE:
            return "H5T_REFERENCE";
            break;
        case H5T_ENUM:
            return "H5T_ENUM";
            break;
        case H5T_VLEN:
            return "H5T_VLEN";
            break;
        case H5T_ARRAY:
            return "H5T_ARRAY";
            break;
        default:
            return "Unknown data type";
            break;
    }
}




// Get the names of all the attributes contained in the current dataset
std::vector<std::string> isce::io::IDataSet::getAttrs() {

    // Initialize container that will contain the attribute names (if any)
    std::vector<std::string> outList;

    // If none, job done, return
    if (H5::H5Object::getNumAttrs() == 0)
        return outList;

    // Iterate over all the attributes and get their names
    H5::H5Object::iterateAttrs(attrsNames , NULL, &outList);
    //int idx = H5::H5Object::iterateAttrs(attrsNames , NULL, &outList);

    return outList;
}



// Get the dataset storage chunk size.
// The returned vector is the same size as the rank of the dataset and contains
// the size of the chunk in each dimensions. A size of 0 means no chunking in 
// that dimension (TODO To be confirmed)
std::vector<int> isce::io::IDataSet::getChunkSize() {

    // First, get the rank of the dataset
    int rank = getRank();

    // Initialize the output container
    std::vector<int> out(rank,0);

    // Get the dataset creation properties list
    H5::DSetCreatPropList plist = getCreatePlist();

    // Check that the dataset is chunked
    if (H5D_CHUNKED == plist.getLayout()) {
        
       // Initialize array to get chunk size
       hsize_t * chunkDims = new hsize_t[rank];   //TODO check for success

       // Get the chunk size
       // rankChunk contains the dimensionality of the chunk. If negative, 
       // then function failed.
       plist.getChunk(rank, chunkDims);
       //int rankChunk = plist.getChunk(rank, chunkDims);

       // TODO check for success. If failure, do ?
	
       // Fill in the output vector
       for (int i=0; i<rank; i++)
          out[i] = (int)chunkDims[i];

       // Close obj and release temporary array
       delete [] chunkDims;
       plist.close();

    }
        
    return out;

}


// Get the number of "used" bits of the current dataset or given attributes as
// stored in the file.
// Although the smallest size unit is the byte, the data might be encoded with 
// less (<8bits). This function return the precision (in bits) of the 
// dataset/attribute. 
int isce::io::IDataSet::getNumBits(const std::string &v) {

    H5::DataType dtype;

    // Looking for the dataset itself
    if (v.empty()) 
       // Open the dataspace of the current dataset
       dtype = H5::DataSet::getDataType();

    // Looking for the attribute of name v contained in the dataset itself
    else if (attrExists(v)){    
        // Open the attribute of the given name
        H5::Attribute attr = openAttribute(v);

        // Open the dataspace of the current attribute
        dtype = attr.getDataType();
        
       // Close the attribute
       attr.close();
    }
    else {}//TODO throw exception

    // Note: from there, the C interface is used as it seems that the C++ 
    // interface does not have all the functionalities implemented
    
    // Get the id (for C interface) of the dataType object
    hid_t dt = dtype.getId();

    // Get the precision in bit of the datatype. 
    size_t precision = H5Tget_precision(dt);

    dtype.close();
    H5Tclose(dt);

    return (int) precision;
};





// TODO: All "read" functions below need some templating to tighten the code


// Function to read an scalar dataset or attribute from file to a variable. 
// Input variable that will receive the data can only be substituted to a 
// numeric type. 
// For dataset/attribute containing more than one element, a vector<T> should
// be passed - see other signature of this function below.
template<typename T>
void isce::io::IDataSet::read(T &v, const std::string &att) {

    // Check that the parameter that will receive the data read from the file 
    // is a numeric variable
    //if (typeid(T) != typeid(isce::core::FixedString))
    //    static_assert(std::is_arithmetic<T>::value, "Expected a numeric scalar"); 

    // Check that the dataset/attribute contains a scalar, i.e., rank=0
    if (getRank(att) != 0) {
        return;   // TODO Should add a message or something here
    }

    if (att.empty()) {
       H5::DataSet::read(&v, memType(v,*this));
    } else if (attrExists(att)) {
       // Open the attribute
       H5::Attribute a = openAttribute(att);

       // Read the attribute from file
       a.read(memType(v,a), &v);  // Note: order of parameter is reversed 
                                   //       compared to dataset read
       // Close the attribute
       a.close();
    }
    else {}   // TODO Should warn or throw here

    return;
}

template void
isce::io::IDataSet::
read(double & v, const std::string & att);

template void
isce::io::IDataSet::
read(float & v, const std::string & att);

template void
isce::io::IDataSet::
read(int & v, const std::string & att);

template void
isce::io::IDataSet::
read(isce::core::FixedString & v, const std::string & att);

// Same as above but for std::string. 
// TODO: Should be templated with above function. Will probably need some
// sort of function specialization as the "read" function called inside is 
// overloaded with parameters being a std::string or void *buffer.
void isce::io::IDataSet::read(std::string &v, const std::string &att) {

    // Check that the dataset/attribute contains a scalar, i.e., rank=0
    if (getRank(att) != 0)
        return;   // TODO Should add a message or something here

    if (att.empty()) {
       H5::DataSet::read(&v, memType(v,*this), getDataSpace());
    }
    else if (attrExists(att)) {
       // Open the attribute
       H5::Attribute a = openAttribute(att);

       // Read the attribute from file
       a.read(a.getStrType(), v);  // Note: order of parameter is reversed 
                                    //       compared to dataset read
       // Close the attribute
       a.close();
    }
    else {}   // TODO Should add a message or something here

    return;
}






template<typename T>
void isce::io::IDataSet::read(T * buffer, const int * startIn, 
                                const int * countIn, 
                                const int * strideIn) { 

    // Format a dataspace according to the input parameters       
    H5::DataSpace dspace = getReadDataSpace(startIn, countIn, strideIn);	 

    // Check that the selection is valid (no out of bound)
    // TODO - to be adjusted with preferred error/message handling
    if (!dspace.selectValid()){
       std::cout << "Sub-selection of dataset is invalid" << std::endl;
       dspace.close();
       return;
    }

    // Get total number of elements to read
    hssize_t nbElements = dspace.getSelectNpoints();
    
    // Format the dataspace of the memory to receive the data read from file
    H5::DataSpace memspace = getReadMemorySpace((hsize_t)nbElements);

    // Read the dataset to memory
    H5::DataSet::read(buffer, memType(buffer,*this) , memspace, dspace);
    
    // Close dataset and memory dataspaces
    dspace.close();
    memspace.close();

}

template<typename T>
void isce::io::IDataSet::read(std::vector<T> &buffer) {
   read(buffer, nullptr, nullptr, nullptr);
} 

template void
isce::io::IDataSet::
read(std::vector<int> & buffer);

template void
isce::io::IDataSet::
read(std::vector<float> & buffer);

template void
isce::io::IDataSet::
read(std::vector<double> & buffer);

template void
isce::io::IDataSet::
read(std::vector<std::string> & buffer);

template void
isce::io::IDataSet::
read(std::vector<isce::core::FixedString> & buffer);


template<typename T>
void isce::io::IDataSet::read(std::vector<T> &buffer, const std::vector<int> * startIn, 
                                            const std::vector<int> * countIn, 
                                            const std::vector<int> * strideIn) { 
 
    // Format a dataspace according to the input parameters       
    H5::DataSpace dspace = getReadDataSpace((startIn)  ? startIn->data() : nullptr,
                                            (countIn)  ? countIn->data() : nullptr,
                                            (strideIn) ? strideIn->data() : nullptr);

    // Check that the selection is valid (no out of bound)
    // TODO - to be adjusted with preferred error/message handling
    if (!dspace.selectValid()){
       std::cout << "Sub-selection of dataset is invalid" << std::endl;
       dspace.close();
       return;
    }

    // Get total number of elements to read
    hssize_t nbElements = dspace.getSelectNpoints();
    
    // Format the dataspace of the memory to receive the data read from file
    H5::DataSpace memspace = getReadMemorySpace((hsize_t)nbElements);

    // Size the output container accordingly to the number of elements to be read
    if(buffer.size() < nbElements)
        buffer.resize(nbElements);

    // Read the dataset to memory
    H5::DataSet::read(buffer.data(), memType(buffer,*this) , memspace, dspace);
    
    // Close dataset and memory dataspaces
    dspace.close();
    memspace.close();

}


template<typename T>
void isce::io::IDataSet::read(std::valarray<T> &buffer) {
   read(buffer, nullptr, nullptr, nullptr);
} 

template<typename T>
void isce::io::IDataSet::read(std::valarray<T> &buffer, const std::valarray<int> *startIn, 
                                              const std::valarray<int> *countIn, 
                                              const std::valarray<int> *strideIn) { 
 
    // Format a dataspace according to the input parameters       
    H5::DataSpace dspace = getReadDataSpace((startIn)  ? &(*startIn)[0] : nullptr,
                                            (countIn)  ? &(*countIn)[0] : nullptr,
                                            (strideIn) ? &(*strideIn)[0] : nullptr);

    // Check that the selection is valid (no out of bound)
    // TODO - to be adjusted with preferred error/message handling
    if (!dspace.selectValid()){
       std::cout << "Sub-selection of dataset is invalid" << std::endl;
       dspace.close();
       return;
    }

    // Get total number of elements to read
    hssize_t nbElements = dspace.getSelectNpoints();
    
    // Format the dataspace of the memory to receive the data read from file
    H5::DataSpace memspace = getReadMemorySpace((hsize_t)nbElements);

    // Size the output container accordingly to the number of elements to be read
    if(buffer.size() < nbElements)
        buffer.resize(nbElements);

    // Read the dataset to memory
    H5::DataSet::read(&buffer[0], memType(buffer,*this) , memspace, dspace);
    
    // Close dataset and memory dataspaces
    dspace.close();
    memspace.close();

}



H5::DataSpace isce::io::IDataSet::getReadMemorySpace(hsize_t nbElements) {

    // Create a memory dataSpace describing the dataspace of the memory buffer
    // that will receive the data. It's set up as a 1D array.
    hsize_t memDims[1] = {(hsize_t)nbElements};
    H5::DataSpace memspace((hsize_t)1,memDims);
    return memspace;

}


H5::DataSpace isce::io::IDataSet::getReadDataSpace(const int * startIn, 
                                                   const int * countIn, 
                                                   const int * strideIn) { 
 
    // Get information of the file dataspace       
    H5::DataSpace dspace = H5::DataSet::getSpace();	 
    int rank = dspace.getSimpleExtentNdims();

    // Specific case if dataset contains a scalar
    if (rank == 0)
        return dspace;


    // HDF5 library expect startIn/countIn/strideIn to be in unsigned long long.
    // Convert input (int) and set default values
    
    // Get information on the start location
    hsize_t *start = new hsize_t[rank];
    if (startIn) {
        for(int i=0; i<rank; i++)
             start[i] = (hsize_t)startIn[i];
    }
    else {
        for (int i=0; i<rank; i++)
            start[i] = 0;
    }

    // Get information on the number of elements to read
    hsize_t *count = new hsize_t[rank];
    if (countIn) {
        for (auto i=0; i<rank; i++)
            count[i] = (hsize_t)countIn[i];
    }
    else
        dspace.getSimpleExtentDims(count, NULL);


    // Get information on the stride
    hsize_t *stride = new hsize_t[rank];
    if (strideIn) {
        for (int i=0; i<rank; i++)
             stride[i] = (hsize_t)strideIn[i];
    }
    else {
        for (int i=0; i<rank; i++)
            stride[i] = 1;
    }

    // Select the subset of the dataset to read
    // Note: H5 throws an error if this function is applied on a scalar dataset
    dspace.selectHyperslab(H5S_SELECT_SET, count, start, stride);

    delete [] start;
    delete [] count;
    delete [] stride;

    return dspace;
}







// Function to read a non-scalar (i.e., multi dimensional) attribute from file 
// to raw pointer. The input parameters are the raw pointer that will hold the
// attribute values and the attribute name. 
// Note: this function is similar to the dataset read method except that 
// subselection is not possible and it needs an attribute name.
template<typename T>
void isce::io::IDataSet::read(T *buffer, const std::string &att) { 

    // Check that attribute name is not null or empty
    // TODO Should add a message or something here
    if (!attrExists(att))
        return;

    // Open the attribute
    H5::Attribute a = openAttribute(att);

    // Read the dataset to memory
    a.read(memType(buffer,a), buffer);

    // Close the attribute
    a.close();
}






// Function to read a non-scalar (i.e., multi dimensional) attribute from file 
// to memory vector<T>. The input parameters are the vector that will hold the
// attribute values and the attribute name. 
// Note: this function is similar to the dataset read method except that 
// subselection is not possible and it needs an attribute name.
template<typename T>
void isce::io::IDataSet::read(std::vector<T> &buffer, const std::string &att) { 

    // Check that attribute name is not null or empty
    // TODO Should add a message or something here
    if (!attrExists(att))
        return;

    // Get number of elements in that attribute       
    int nbElements = getNumElements(att);

    // Open the attribute
    H5::Attribute a = openAttribute(att);

    // Size the output container accordingly to the number of elements read
    if(buffer.size() < nbElements)
        buffer.resize(nbElements);

    // Read the dataset to memory
    a.read(memType(buffer,a), buffer.data());

    // Close the attribute
    a.close();
}



// Function to read a non-scalar (i.e., multi dimensional) attribute from file 
// to memory vector<T>. The input parameters are the vector that will hold the
// attribute values and the attribute name. 
// Note: this function is similar to the previous one except that subselection
// is not possible and it needs an attribute name.
template<typename T>
void isce::io::IDataSet::read(std::valarray<T> &buffer, const std::string &att) { 

    // Check that attribute name is not null or empty
    // TODO Should add a message or something here
    if (!attrExists(att))
        return;

    // Get number of elements in that attribute       
    int nbElements = getNumElements(att);

    // Open the attribute
    H5::Attribute a = openAttribute(att);

    // Size the output container accordingly to the number of elements read
    if(buffer.size() < nbElements)
        buffer.resize(nbElements);

    // Read the dataset to memory
    a.read(memType(buffer,a), &buffer[0]);

    // Close the attribute
    a.close();
}




// Open an IDataset object
isce::io::IDataSet isce::io::IH5File::openDataSet(const H5std_string &name) {
            
    H5::DataSet dset = H5::H5File::openDataSet(name);
    return IDataSet(dset);

};



// This function is a search function allowing the user to list (in a 
// vector<string> container) all the group and dataset whose name 
// (i.e. path/name) contains a given string.
// IN: nameIn: String to be searched in the file tree
// IN start (default: "/"): Location at which the search is to be started. That is
// the function will search that location and all subdirectories.
// IN type (default: "BOTH"): Whether to return only DATASET, GROUP or BOTH
// Note that whether or not the user pass in a start path for the search, the 
// returned paths of the objects founds are always the full absolute path.

std::vector<std::string> isce::io::IH5File::find(std::string nameIn, std::string start,
                                       std::string type) {

    std::vector<std::string> outList;

    if (nameIn.empty())
        return outList;

    if (start != "/") {
        // Check that the provided start path exists in the file
        if (!this->exists(start))
            return outList;

        // Check that the provided start path points to a group
        H5O_info_t *info = nullptr;
        herr_t status = H5Oget_info_by_name(this->getId(), start.c_str(), 
                                                    info, H5P_DEFAULT);
        if (status < 0 || info->type != H5O_TYPE_GROUP)
           return outList;
    }

    if (type != "GROUP" && type != "DATASET" && type != "BOTH")
        return outList;


    //Hack. Store search name and type in outList as one user parameter is allowed
    outList.push_back(nameIn);
    outList.push_back(type);
    // Some pre-process of "start" to allow full path return
    if (start.length() > 1 && strncmp(&start.back(),"/",1))
        outList.push_back(start.append("/"));
    else 
        outList.push_back(start);
  

    H5Ovisit_by_name(this->getId(), start.c_str(), H5_INDEX_NAME, H5_ITER_INC, 
                     matchName, &outList, H5P_DEFAULT);


    //Hack. Removing the search name and type from outList.
    outList.erase(outList.begin(), outList.begin()+3);

    return outList; 

}





