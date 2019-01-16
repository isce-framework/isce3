//

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <gtest/gtest.h>

#include "isce/io/IH5.h"

// Support function to check if file exists
inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
};





struct IH5Test : public ::testing::Test {

   isce::io::IH5File file;

   // Constructor
    protected:
        IH5Test() : file("../../data/envisat.h5"){
        }

};



TEST_F(IH5Test, findInFile) {

    // Name to search for in the file
    std::string searchedName("hh");
    // Location where to start a search
    std::string startName("/science/metadata/calibration_info");
    // Root character
    char root = '/';

    // Run the default find function, i.e.:
    // - search from root ("/")
    // - Group and Dataset
    // - return absolute path
    std::vector<std::string> list = file.find(searchedName);
    ASSERT_EQ(list.size(), 25);
    for (auto i:list)
       ASSERT_TRUE(i[0] == root);

    // Clear list
    list.clear();

    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - search from /science/complex_imagery
    // - Group and Dataset
    // - return ABSOLUTE path
    list = file.find(searchedName, startName);
    ASSERT_EQ(list.size(), 14);
    for (auto i:list)
       ASSERT_TRUE(i[0] == root);

    // Clear list
    list.clear();

    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Group and Dataset
    // - return RELATIVE path
    list = file.find(searchedName, startName, "BOTH", "RELATIVE");
    ASSERT_EQ(list.size(), 14);
    for (auto i:list)
       ASSERT_FALSE(i[0] == root);

    // Clear list
    list.clear();

    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Dataset only
    // - return relative path
    list = file.find(searchedName, startName, "DATASET", "RELATIVE");
    ASSERT_EQ(list.size(), 12);
    for (auto i:list)
       ASSERT_FALSE(i[0] == root);

        
    // Clear list
    list.clear();


    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Group only
    // - return full path
    list = file.find(searchedName, startName, "GROUP", "FULL");
    ASSERT_EQ(list.size(), 2);
    for (auto i:list)
       ASSERT_TRUE(i[0] == root);
    for (auto i:list) {
        hid_t id = H5Oopen(file.getId(), i.c_str(), H5P_DEFAULT);
        ASSERT_EQ(H5Iget_type(id), H5I_GROUP);
        H5Oclose(id);
    }

}




TEST_F(IH5Test, findInGroup) {

    // Group name
    std::string groupName("/science/metadata");
    // Name to search for in the group
    std::string searchedName("hh");
    // Location in the group where to start the search
    std::string startName("calibration_info");


    // Open group
    isce::io::IGroup group = file.openGroup(groupName);


    // Run the default find function on the group, i.e.:
    // - search from group root
    // - Group and Dataset
    // - return absolute path from Group location
    std::vector<std::string> list = group.find(searchedName);
    ASSERT_EQ(list.size(), 18);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c' || i[0] == 'n');

    // Clear list
    list.clear();

    // Run the find function from the startName location in the group
    // - search from startName in the group location
    // - Group and Dataset
    // - return ABSOLUTE path (i.e., from group root, not file root "/")
    list = group.find(searchedName, startName);
    ASSERT_EQ(list.size(), 14);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c');

    // Clear list
    list.clear();

    // Run the find function from the startName in thegroup
    // - search from startName
    // - Group and Dataset
    // - return RELATIVE path (i.e., from startName, not from group root)
    list = group.find(searchedName, startName, "BOTH", "RELATIVE");
    ASSERT_EQ(list.size(), 14);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'a' || i[0] == 'c');

    // Clear list
    list.clear();

    // Run the find function from the startName in the group
    // - search from startName
    // - Dataset only
    // - return relative path
    list = group.find(searchedName, startName, "DATASET", "RELATIVE");
    ASSERT_EQ(list.size(), 12);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c' || i[0] == 'a');
//    for (auto i:list) {
//        isce::io::IDataSet dset = group.openDataSet(i);   
//        ASSERT_EQ(dset.getId(), H5I_DATASET);
//        dset.close();
//    }
        
    // Clear list
    list.clear();

    // Run the find function from the startName in the 
    // - search from startName
    // - Group only
    // - return full path
    list = group.find(searchedName, startName, "GROUP", "FULL");
    ASSERT_EQ(list.size(), 2);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c' || i[0] == 'a');
    for (auto i:list) {
        //isce::io::IDataSet dset = group.openDataSet(i);   
        //ASSERT_EQ(dset.getId(), H5I_DATASET);
        //dset.close();
        hid_t id = H5Oopen(group.getId(), i.c_str(), H5P_DEFAULT);
        ASSERT_EQ(H5Iget_type(id), H5I_GROUP);
        H5Oclose(id);
    }

    group.close();

}




TEST_F(IH5Test, findRegularExpression) {

    // Regular expression to search for in the file
    std::string searchedName(".*mode.*(hh$|hv$|rh$|rv$|vh$|vv$)");

    // Run the default find function, i.e.:
    // - search from root ("/")
    // - Group and Dataset
    // - return absolute path
    std::vector<std::string> list = file.find(searchedName);
    ASSERT_EQ(list.size(), 26);
    for (auto i:list)
       ASSERT_TRUE(i[0] == '/');

}





TEST_F(IH5Test, dataSetOpen) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");

    isce::io::IDataSet dset = file.openDataSet(datasetName);

    // Check that the type of the open object corresponds to a dataset type
    ASSERT_EQ(H5Iget_type(dset.getId()), H5I_DATASET);

    dset.close();
}



TEST_F(IH5Test, dataSetMetaData) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");

    isce::io::IDataSet dset = file.openDataSet(datasetName);

    // Nb dimensions
    ASSERT_EQ(dset.getRank(), 2);

    // Dimensions 
    std::vector<int> dims = dset.getDimensions();
    ASSERT_EQ(dims[0], 500); 
    ASSERT_EQ(dims[1], 500); 

    // Number of elements
    ASSERT_EQ(dset.getNumElements(), 250000);

    // Dataset values type
    ASSERT_EQ(dset.getTypeClassStr(), "H5T_COMPOUND");

    // Dataset number of attributes
    ASSERT_EQ(dset.getNumAttrs(), 3);

    //Get attributes names
    std::vector<std::string> attnames = dset.getAttrs();
    ASSERT_EQ(attnames.size(), 3);
    ASSERT_EQ(attnames[0], "description");
    ASSERT_EQ(attnames[1], "spacing");
    ASSERT_EQ(attnames[2], "units");

    dset.close();
}


// Check for attribute metadata reading
TEST_F(IH5Test, attributeMetaData) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    std::string attributeName("description");

    isce::io::IDataSet dset = file.openDataSet(datasetName);

    ASSERT_EQ(dset.getRank(attributeName), 0); //scalar attribute
    ASSERT_EQ(dset.getNumElements(attributeName), 1);
    ASSERT_EQ(dset.getTypeClassStr(attributeName),"H5T_STRING");

    dset.close();
}


//Testing reading string variable length in a std::string
TEST_F(IH5Test, readVariableLengthString) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    std::string attribute("description");
    isce::io::IDataSet dset = file.openDataSet(datasetName);
    

    std::string strVal;
    dset.read(strVal, attribute);

    ASSERT_EQ(strVal, "Complex backscatter for primary mode (HH pol). Focused SLC image. All channels are registered");

    dset.close();
}

/* 

//Testing reading string variable length in a char *
TEST_F(IH5Test, readVariableLengthString2) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    std::string attribute("description");
    isce::io::IDataSet dset = file.openDataSet(datasetName);
    

    char *strVal;
    dset.read(strVal, attribute);

    ASSERT_EQ(strVal, "Complex backscatter for primary mode (HH pol). Focused SLC image. All channels are registered");

    dset.close();
}

*/



//Testing reading string fixed length
TEST_F(IH5Test, readFixedLengthString) {

    std::string datasetName("/science/complex_imagery/aux_mode/zero_doppler_start_az_time");
    isce::io::IDataSet dset = file.openDataSet(datasetName);
    
    std::string strVal;
    dset.read(strVal);

    ASSERT_EQ(strVal, "2003-02-26T17:55:30.843491759");

    dset.close();
}





//Testing reading dataset with raw pointer
TEST_F(IH5Test, datasetReadComplexWithRawPointer) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    isce::io::IDataSet dset = file.openDataSet(datasetName);

    // Dataset is a 500x500 floating complex image

    std::complex<float> * dval = new (std::nothrow) std::complex<float>[500*500];
    dset.read(dval);
    ASSERT_FLOAT_EQ(dval[3].real(), 2.01395082);
    ASSERT_FLOAT_EQ(dval[3].imag(), 2.99577761);

    ASSERT_FLOAT_EQ(dval[503].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[503].imag(), -0.007895827);

    ASSERT_FLOAT_EQ(dval[249999].real(), 0.41759312);
    ASSERT_FLOAT_EQ(dval[249999].imag(), -0.62631774);

    delete [] dval;

    // Partial read of dataset: (3,1)->(10, 11)

    // Partial read using HDF5-like subsetter (start, count, stride)
    int *startIn  = new int[2];
    int *countIn  = new int[2];
    int *strideIn = new int[2];
    startIn[0] = 1;
    startIn[1] = 3;
    countIn[0] = 11;
    countIn[1] = 8;
    strideIn[0] = 1;
    strideIn[1] = 1;

    dval = new (std::nothrow) std::complex<float>[11*8];
    dset.read(dval, startIn, countIn, strideIn);
    
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);

    delete [] dval;
    delete [] startIn;
    delete [] countIn;
    delete [] strideIn;


    // Partial read using std::slice subsetter
    std::vector<std::slice> vslices(2);
    vslices[0] = std::slice(1,11,1);
    vslices[1] = std::slice(3,8,1);

    dval = new (std::nothrow) std::complex<float>[11*8];
    dset.read(dval, &vslices);

    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);

    delete [] dval;


    // Partial read using std::gslice subsetter
    std::gslice gslices(503, {11,8}, {500,1});

    dval = new (std::nothrow) std::complex<float>[11*8];
    dset.read(dval, &gslices);

    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);

    delete [] dval;

    dset.close();

}






//Testing reading dataset with std::vector
TEST_F(IH5Test, datasetReadComplexWithVector) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    isce::io::IDataSet dset = file.openDataSet(datasetName);

    // Dataset is a 500x500 floating complex image

    std::vector<std::complex<float>> dval;
    dset.read(dval);
    ASSERT_EQ(dval.size(), 250000); // 
    ASSERT_FLOAT_EQ(dval[3].real(), 2.01395082);
    ASSERT_FLOAT_EQ(dval[3].imag(), 2.99577761);

    ASSERT_FLOAT_EQ(dval[503].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[503].imag(), -0.007895827);

    ASSERT_FLOAT_EQ(dval[249999].real(), 0.41759312);
    ASSERT_FLOAT_EQ(dval[249999].imag(), -0.62631774);


    // Partial read of dataset: (3,1)->(10, 11)
 
    // Partial read using HDF5-like subsetter (start, count, stride)
    std::vector<int> startIn, countIn, strideIn;
    startIn.push_back(1);
    startIn.push_back(3);
    countIn.push_back(11);
    countIn.push_back(8);
    strideIn.push_back(1);
    strideIn.push_back(1);

    dval.clear();
    dset.read(dval, &startIn, &countIn, &strideIn);
    
    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);



    // Partial read using std::slice subsetter
    std::vector<std::slice> vslices(2);
    vslices[0] = std::slice(1,11,1);
    vslices[1] = std::slice(3,8,1);

    dval.clear();
    dset.read(dval, &vslices);

    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);


    // Partial read using std::gslice subsetter
    std::gslice gslices(503, {11,8}, {500,1});

    dval.clear();
    dset.read(dval, &gslices);

    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);

    dset.close();

}




//Testing reading dataset with std::valarray
TEST_F(IH5Test, datasetReadComplexWithValarray) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    isce::io::IDataSet dset = file.openDataSet(datasetName);

    // Dataset is a 500x500 floating complex image

    std::valarray<std::complex<float>> dval;
    dset.read(dval);
    ASSERT_EQ(dval.size(), 250000); // 
    ASSERT_FLOAT_EQ(dval[3].real(), 2.01395082);
    ASSERT_FLOAT_EQ(dval[3].imag(), 2.99577761);

    ASSERT_FLOAT_EQ(dval[503].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[503].imag(), -0.007895827);

    ASSERT_FLOAT_EQ(dval[249999].real(), 0.41759312);
    ASSERT_FLOAT_EQ(dval[249999].imag(), -0.62631774);


    // Partial read of dataset: (3,1)->(10, 11)
 
    // Partial read using HDF5-like subsetter (start, count, stride)
    std::valarray<int> startIn({1,3}), countIn({11,8}), strideIn({1,1});

    std::valarray<std::complex<float>>().swap(dval); //freeing memory
    dset.read(dval, &startIn, &countIn, &strideIn);
    
    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);



    // Partial read using std::slice subsetter
    std::vector<std::slice> vslices(2);
    vslices[0] = std::slice(1,11,1);
    vslices[1] = std::slice(3,8,1);

    std::valarray<std::complex<float>>().swap(dval); //freeing memory
    dset.read(dval, &vslices);

    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);


    // Partial read using std::gslice subsetter
    std::gslice gslices(503, {11,8}, {500,1});

    std::valarray<std::complex<float>>().swap(dval); //freeing memory
    dset.read(dval, &gslices);

    ASSERT_EQ(dval.size(), 88);
    ASSERT_FLOAT_EQ(dval[0].real(), -1.3640736);
    ASSERT_FLOAT_EQ(dval[0].imag(), -0.007895827);
    ASSERT_FLOAT_EQ(dval[87].real(), 5.6398234);
    ASSERT_FLOAT_EQ(dval[87].imag(), -9.047101);

    dset.close();

}



//Testing reading 8-bit unsigned integer dataset with raw pointer
//TODO: Empty fields in dataset at time of writing. Assertion uncertain
TEST_F(IH5Test, datasetReadU8WithRawPointer) {

    std::string datasetName("/science/image_quality_flags/missinglines");
    isce::io::IDataSet dset = file.openDataSet(datasetName);

    std::vector<unsigned char> dval;
    dset.read(dval);
    ASSERT_EQ(dval[0], 0);

    dset.close();

}


//Testing reading float dataset with vector
//TODO: Empty fields in dataset at time of writing. Assertion uncertain
TEST_F(IH5Test, datasetReadFloatWithVector) {

    std::string datasetName("/science/metadata/attitude/predict/angular_velocity");
    isce::io::IDataSet dset = file.openDataSet(datasetName);

    std::vector<float> dval;
    dset.read(dval);
    ASSERT_EQ(dval.size(), 33);
    ASSERT_FLOAT_EQ(dval[0], 0);

    dset.close();
}

// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
