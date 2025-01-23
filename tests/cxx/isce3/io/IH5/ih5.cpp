//

#include <numeric>
#include <gtest/gtest.h>

#include "isce3/io/IH5.h"


// Output HDF5 file for file creation/writing tests
std::string wFileName("dummyHdf5.h5");
std::string rFileName(TESTDATA_DIR "envisat.h5");

struct IH5Test : public ::testing::Test {

   isce3::io::IH5File file;

   // Constructor
    protected:
        IH5Test() : file(rFileName){
        }

};




/* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 *                                         IH5 API Reading test
 * = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
*/



TEST_F(IH5Test, findInFile) {

    // Name to search for in the file
    std::string searchedName("HH");
    // Location where to start a search
    std::string startName("/science/LSAR/SLC/metadata/calibrationInformation");
    // Root character
    char root = '/';

    // Run the default find function, i.e.:
    // - search from root ("/")
    // - Group and Dataset
    // - return absolute path
    std::vector<std::string> list = file.find(searchedName);
    ASSERT_EQ(list.size(), 5);
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
    ASSERT_EQ(list.size(), 4);
    for (auto i:list)
       ASSERT_TRUE(i[0] == root);

    // Clear list
    list.clear();

    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Group and Dataset
    // - return RELATIVE path
    list = file.find(searchedName, startName, "BOTH", "RELATIVE");
    ASSERT_EQ(list.size(), 4);
    for (auto i:list)
       ASSERT_FALSE(i[0] == root);

    // Clear list
    list.clear();

    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Dataset only
    // - return relative path
    list = file.find(searchedName, startName, "DATASET", "RELATIVE");
    ASSERT_EQ(list.size(), 3);
    for (auto i:list)
       ASSERT_FALSE(i[0] == root);

        
    // Clear list
    list.clear();


    // Run the find function from a group and its subgroups only
    // - search from /science/metadata/calibration_info
    // - Group only
    // - return full path
    list = file.find(searchedName, startName, "GROUP", "FULL");
    ASSERT_EQ(list.size(), 1);
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
    std::string groupName("/science/LSAR/SLC/metadata");
    // Name to search for in the group
    std::string searchedName("HH");
    // Location in the group where to start the search
    std::string startName("calibrationInformation");


    // Open group
    isce3::io::IGroup group = file.openGroup(groupName);


    // Run the default find function on the group, i.e.:
    // - search from group root
    // - Group and Dataset
    // - return absolute path from Group location
    std::vector<std::string> list = group.find(searchedName);
    ASSERT_EQ(list.size(), 4);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c' || i[0] == 'n');

    // Clear list
    list.clear();

    // Run the find function from the startName location in the group
    // - search from startName in the group location
    // - Group and Dataset
    // - return ABSOLUTE path (i.e., from group root, not file root "/")
    list = group.find(searchedName, startName);
    ASSERT_EQ(list.size(), 4);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c');

    // Clear list
    list.clear();

    // Run the find function from the startName in thegroup
    // - search from startName
    // - Group and Dataset
    // - return RELATIVE path (i.e., from startName, not from group root)
    list = group.find(searchedName, startName, "BOTH", "RELATIVE");
    ASSERT_EQ(list.size(), 4);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'f');

    // Clear list
    list.clear();

    // Run the find function from the startName in the group
    // - search from startName
    // - Dataset only
    // - return relative path
    list = group.find(searchedName, startName, "DATASET", "RELATIVE");
    ASSERT_EQ(list.size(), 3);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'f');
//    for (auto i:list) {
//        isce3::io::IDataSet dset = group.openDataSet(i);   
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
    ASSERT_EQ(list.size(), 1);
    for (auto i:list)
       ASSERT_TRUE(i[0] == 'c' || i[0] == 'a');
    for (auto i:list) {
        //isce3::io::IDataSet dset = group.openDataSet(i);   
        //ASSERT_EQ(dset.getId(), H5I_DATASET);
        //dset.close();
        hid_t id = H5Oopen(group.getId(), i.c_str(), H5P_DEFAULT);
        ASSERT_EQ(H5Iget_type(id), H5I_GROUP);
        H5Oclose(id);
    }
}




TEST_F(IH5Test, findRegularExpression) {

    // Regular expression to search for in the file
    std::string searchedName(".*calibrationInformation.*(HH$|HV$|RH$|RV$|VH$|VV$)");

    // Run the default find function, i.e.:
    // - search from root ("/")
    // - Group and Dataset
    // - return absolute path
    std::vector<std::string> list = file.find(searchedName);
    ASSERT_EQ(list.size(), 7);
    for (auto i:list)
       ASSERT_TRUE(i[0] == '/');

}





TEST_F(IH5Test, dataSetOpen) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");

    isce3::io::IDataSet dset = file.openDataSet(datasetName);

    // Check that the type of the open object corresponds to a dataset type
    ASSERT_EQ(H5Iget_type(dset.getId()), H5I_DATASET);
}



TEST_F(IH5Test, dataSetMetaData) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");

    isce3::io::IDataSet dset = file.openDataSet(datasetName);

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
    ASSERT_EQ(dset.getNumAttrs(), 1);

    //Get attributes names
    std::vector<std::string> attnames = dset.getAttrs();
    ASSERT_EQ(attnames.size(), 1);
    ASSERT_EQ(attnames[0], "units");
}


// Check for attribute metadata reading
TEST_F(IH5Test, attributeMetaData) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");
    std::string attributeName("units");

    isce3::io::IDataSet dset = file.openDataSet(datasetName);

    ASSERT_EQ(dset.getRank(attributeName), 0); //scalar attribute
    ASSERT_EQ(dset.getNumElements(attributeName), 1);
    ASSERT_EQ(dset.getTypeClassStr(attributeName),"H5T_STRING");
}


//Testing reading string variable length in a std::string
TEST_F(IH5Test, readVariableLengthString) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");
    std::string attribute("units");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);
    

    std::string strVal;
    dset.read(strVal, attribute);

    ASSERT_EQ(strVal, "DN");
}

/* 

//Testing reading string variable length in a char *
TEST_F(IH5Test, readVariableLengthString2) {

    std::string datasetName("/science/complex_imagery/primary_mode/hh");
    std::string attribute("description");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);
    

    char *strVal;
    dset.read(strVal, attribute);

    ASSERT_EQ(strVal, "Complex backscatter for primary mode (HH pol). Focused SLC image. All channels are registered");
}

*/



//Testing reading string variable length
TEST_F(IH5Test, readVariableLengthString2) {

    std::string datasetName("/science/LSAR/identification/boundingPolygon");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);
    
    std::string strVal;
    dset.read(strVal);

    ASSERT_EQ(strVal, "POLYGON ((-115.507 34.822, -115.634 34.845, -115.639 34.827, -115.512 34.805, -115.507 34.822))");
}





//Testing reading dataset with raw pointer
TEST_F(IH5Test, datasetReadComplexWithRawPointer) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);

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
}






//Testing reading dataset with std::vector
TEST_F(IH5Test, datasetReadComplexWithVector) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);

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
}




//Testing reading dataset with std::valarray
TEST_F(IH5Test, datasetReadComplexWithValarray) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);

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
}



//Testing reading 16-bit unsigned integer dataset with raw pointer
//TODO: Empty fields in dataset at time of writing. Assertion uncertain
TEST_F(IH5Test, datasetReadU16WithRawPointer) {

    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/validSamples");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);

    std::vector<unsigned int> dval;
    dset.read(dval);
    ASSERT_EQ(dval[0], 0);
    ASSERT_EQ(dval[1], 500);
}


//Testing reading float dataset with vector
//TODO: Empty fields in dataset at time of writing. Assertion uncertain
TEST_F(IH5Test, datasetReadFloatWithVector) {

    std::string datasetName("/science/LSAR/SLC/metadata/attitude/angularVelocity");
    isce3::io::IDataSet dset = file.openDataSet(datasetName);

    std::vector<float> dval;
    dset.read(dval);
    ASSERT_EQ(dval.size(), 33);
    ASSERT_FLOAT_EQ(dval[0], 0);
}




/* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 *                                         IH5 API Writing test
 * = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
*/



TEST_F(IH5Test, openFileMode) {

    // Delete file if it exists
    struct stat buffer;
    if (stat (wFileName.c_str(), &buffer) == 0)
       std::remove(wFileName.c_str());

    isce3::io::IH5File fic;

    // Open unexisting file in default (reading) mode
    EXPECT_ANY_THROW(fic = isce3::io::IH5File(wFileName));

    // Open unexisting file in explicit reading mode
    EXPECT_ANY_THROW(fic = isce3::io::IH5File(wFileName,'r'));

    // Open unexisting file for read/write    
    EXPECT_ANY_THROW(fic = isce3::io::IH5File(wFileName,'w'));
    
    // Create file for write - whether or not it exists    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'x'));
    fic.close();
  
    // Create file for write - only if it does not exists
    EXPECT_ANY_THROW(fic = isce3::io::IH5File(wFileName,'a'));

    // Delete the empty file
    std::remove(wFileName.c_str());

    // Create file for write - only if it does not exists
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'a'));
}


TEST_F(IH5Test, createGroups) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;

    // Create file for write - whether or not it exists    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'x'));

    // Create a few groups in "/"
    EXPECT_NO_THROW(isce3::io::IGroup grp = fic.createGroup(std::string("groupVector")));
    EXPECT_NO_THROW(isce3::io::IGroup grp = fic.createGroup(std::string("groupValarray")));
    EXPECT_NO_THROW(isce3::io::IGroup grp = fic.createGroup("groupRawPointer"));
    EXPECT_NO_THROW(isce3::io::IGroup grp = fic.createGroup("groupScalar"));

    // Does that group exists?
    list = fic.find(std::string("groupVector"), "/", "GROUP");
    ASSERT_EQ(list.size(), 1);

    // Create a group with unexisiting group path    
    EXPECT_NO_THROW(isce3::io::IGroup grp = fic.createGroup(std::string("groupX/groupXX/groupXXX")));

    // Does that group exists?
    list.clear();
    list = fic.find(std::string("groupXXX"), "/", "GROUP");
    ASSERT_EQ(list.size(), 1);
}



TEST_F(IH5Test, createSimpleDatasetFromVectorBuffer) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));



    // Simple std::vector array with indices values
    std::vector<int> v1(10*10);
    std::iota(v1.begin(), v1.end(), 1);



    // Create simple dataset in groupVector, stored as 1D array from a std::vector buffer
    isce3::io::IGroup grp = fic.openGroup("/groupVector");
    dset = grp.createDataSet(std::string("v1"), v1);

    // Check that Dataset has been created
    list = fic.find("v1","/","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    std::vector<int> v1r;
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 100);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();





    // Create simple dataset, stored as 2D array, from a std::vector buffer
    std::array<int, 2> dims = {10,10};
    dset = grp.createDataSet(std::string("v2"), v1, dims);
    
    // Check that Dataset has been created
    list.clear();
    list = fic.find("v2","/","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Check that rank of dataset is 2, and dimensions are 10x10
    std::vector<int> dims2;
    dims2 = dset.getDimensions();
    ASSERT_EQ(dset.getRank(),2);
    ASSERT_EQ(dims2[0],10);
    ASSERT_EQ(dims2[1],10);

    // Read back the values and check that they are correct
    v1r.clear();
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 100);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();
}


TEST_F(IH5Test, createSimpleDatasetFromValarrayBuffer) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));


    // Simple std::vector array with indices values
    std::valarray<int> v1(10*10);
    int i=1;
    for(auto& v:v1) v=i++;



    // Create simple dataset in groupValarray, stored as 1D array from a std::valarray buffer
    isce3::io::IGroup grp = fic.openGroup("/groupValarray");
    dset = grp.createDataSet(std::string("v1"), v1);

    // Check that Dataset has been created
    list = fic.find("v1","/groupValarray","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    std::valarray<int> v1r;
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 100);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();



    // Create simple dataset, stored as 2D array, from a std::valarray buffer
    std::array<int, 2> dims = {10,10};
    dset = grp.createDataSet(std::string("v2"), v1, dims);
    
    // Check that Dataset has been created
    list.clear();
    list = fic.find("v2","/groupValarray","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Check that rank of dataset is 2, and dimensions are 10x10
    std::vector<int> dims2;
    dims2 = dset.getDimensions();
    ASSERT_EQ(dset.getRank(),2);
    ASSERT_EQ(dims2[0],10);
    ASSERT_EQ(dims2[1],10);

    // Read back the values and check that they are correct
    v1r.resize(0); // clear the valarray
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 100);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();
}


TEST_F(IH5Test, createSimpleDatasetFromRawPointer) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));


    // Simple buffer with indices values
    int* v1 = new int[10*10];
    for(int i=0; i<100; i++) v1[i] = i+1;

    // Create simple dataset in groupRawPointer, stored as 1D array from a std::valarray buffer
    isce3::io::IGroup grp = fic.openGroup("/groupRawPointer");
    dset = grp.createDataSet<int>(std::string("v1"), v1, 100);

    // Check that Dataset has been created
    list = fic.find("v1","/groupRawPointer","DATASET");
    ASSERT_EQ(list.size(), 1);
    ASSERT_EQ(dset.getNumElements(), 100);

    // Read back the values and check that they are correct
    int* v1r = new int[100];
    dset.read(v1r);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();



    // Create simple dataset, stored as 2D array, from a raw pointer
    std::array<int, 2> dims = {10,10};
    dset = grp.createDataSet<int,int>(std::string("v2"), v1, dims);
    
    // Check that Dataset has been created
    list.clear();
    list = fic.find("v2","/groupRawPointer","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Check that rank of dataset is 2, and dimensions are 10x10
    std::vector<int> dims2;
    dims2 = dset.getDimensions();
    ASSERT_EQ(dset.getRank(),2);
    ASSERT_EQ(dims2[0],10);
    ASSERT_EQ(dims2[1],10);

    // Read back the values and check that they are correct
    for(int i=0; i<100; i++) v1r[i]=0;
    dset.read(v1r);
    ASSERT_EQ(v1r[0], 1);
    ASSERT_EQ(v1r[99], 100);
    dset.close();

    delete [] v1;
    delete [] v1r;
}


TEST_F(IH5Test, createSimpleDatasetFromScalar) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));

    float val1 = 9;
    std::string val2("value1");

    // Create simple dataset in groupScalar, stored as an int scalar
    isce3::io::IGroup grp = fic.openGroup("/groupScalar");
    dset = grp.createDataSet(std::string("v1"), val1);

    // Check that Dataset has been created
    list = fic.find("v1", "/groupScalar", "DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    float val1r;
    dset.read(val1r);
    ASSERT_EQ(val1r, 9);

    // Same thing with a string
    dset = grp.createDataSet(std::string("v2"), val2);

    // Check that Dataset has been created
    list.clear();
    list = fic.find("v2", "/groupScalar", "DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    std::string val2r;
    dset.read(val2r);
    ASSERT_EQ(val2r.compare(std::string("value1")), 0);
}



TEST_F(IH5Test, createGroupAttributes) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));


    // Open the root group
    isce3::io::IGroup grp = fic.openGroup("/");

    // Create a scalar attribute (int) in the root group
    int att1 = 9;
    grp.createAttribute(std::string("att1"), att1);

    // Create a scalar attribute (string) in the root group
    //std::string att2("Root Group");
    std::string att2("Root Group attribute");
    grp.createAttribute(std::string("att2"), att2);

    // Check that the attributes have been written
    list = grp.getAttrs();
    ASSERT_EQ(list.size(), 2);

    // Check the values of the attributes
    int att1r;
    grp.read(att1r, list[0]);
    ASSERT_EQ(att1, att1r);

    std::string att2r;
    grp.read(att2r, list[1]);
    ASSERT_EQ(att2.compare(att2r), 0);


    // Close current group and open another one
    grp.close();
    list.clear();
    grp = fic.openGroup("/groupVector");

    // Create a 1D array attribute (float) in the groupVector group
    std::vector<float> attv1;
    for(int i=0; i<10; i++) attv1.push_back(i);
    grp.createAttribute("att1", attv1);

    // Check that the attributes have been written
    list = grp.getAttrs();
    ASSERT_EQ(list.size(), 1);

    
    // Check the values of the attributes
    std::vector<float> attv1r;
    grp.read(attv1r, list[0]);
    for(int i=0; i<attv1.size(); i++)
        ASSERT_EQ(attv1[i], attv1r[i]);
}


TEST_F(IH5Test, createDataSetAttributes) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));

    // Open a group
    isce3::io::IGroup grp = fic.openGroup("/groupVector");

    // Open dataset v1
    dset = grp.openDataSet("v1");

    // Create a scalar attribute (int) in the root group
    int att1 = 9;
    dset.createAttribute(std::string("att1"), att1);

    // Create a scalar attribute (string) in the root group
    //std::string att2("Root Group");
    std::string att2("dataset attribute");
    dset.createAttribute(std::string("att2"), att2);

    // Check that the attributes have been written
    list = dset.getAttrs();
    ASSERT_EQ(list.size(), 2);

    // Check the values of the attributes
    int att1r;
    dset.read(att1r, list[0]);
    ASSERT_EQ(att1, att1r);

    std::string att2r;
    dset.read(att2r, list[1]);
    ASSERT_EQ(att2.compare(att2r), 0);


    // Close current dataset and open another one
    dset.close();
    list.clear();
    dset = grp.openDataSet("v2");

    // Create a 1D array attribute (float) in the groupVector group
    std::vector<float> attv1;
    for(int i=0; i<10; i++) attv1.push_back(i);
    dset.createAttribute("att1", attv1);

    // Check that the attributes have been written
    list = grp.getAttrs();
    ASSERT_EQ(list.size(), 1);

    
    // Check the values of the attributes
    std::vector<float> attv1r;
    dset.read(attv1r, list[0]);
    for(int i=0; i<attv1.size(); i++)
        ASSERT_EQ(attv1[i], attv1r[i]);
}


TEST_F(IH5Test, createFloat16Dataset) {

    std::vector<std::string> list;
    isce3::io::IH5File fic;
    isce3::io::IDataSet dset;

    // Open file for write    
    EXPECT_NO_THROW(fic = isce3::io::IH5File(wFileName,'w'));


    // Fill a vector with 0-255 values
    std::vector<float> v1(1000*1000);
    for(int i=0; i<1000*1000; i++) v1[i] = (float)((i+1)%255);


    ////////////////////////////////////////////////////////////
    // Create a dataset with float16 values, without NBIT filter
    ////////////////////////////////////////////////////////////

    isce3::io::IGroup grp = fic.openGroup("/groupVector");
    std::array<int, 2> dims = {1000,1000};
    dset = grp.createDataSet<isce3::io::float16>(std::string("vFloat16"), dims);
    dset.write(v1);

    // Check that Dataset has been created
    list = fic.find("vFloat16","/","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    std::vector<float> v1r;
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 1000000);
    ASSERT_EQ(v1r[0], v1[0]);
    ASSERT_EQ(v1r[99], v1[99]);

    // Check the size on disk of the dataset
    long storageSize1 = dset.getStorageSize();
    ASSERT_EQ(storageSize1, 4000000);

    dset.close();



    ////////////////////////////////////////////////////////////
    // Create a dataset with float16 values, with NBIT filter
    ////////////////////////////////////////////////////////////

    dset = grp.createDataSet<isce3::io::float16>(std::string("vFloat16_nbit"), dims,1);
    dset.write(v1);

    // Check that Dataset has been created
    list.clear();
    list = fic.find("vFloat16_nbit","/","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    v1r.clear();
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 1000000);
    ASSERT_EQ(v1r[0], v1[0]);
    ASSERT_EQ(v1r[99], v1[99]);

    // Check the size on disk of the dataset
    long storageSize2 = dset.getStorageSize();
    EXPECT_GT(storageSize1, storageSize2);

    dset.close();


    ////////////////////////////////////////////////////////////
    // Create a dataset with float16 values, with NBIT filter + deflate
    ////////////////////////////////////////////////////////////

    dset = grp.createDataSet<isce3::io::float16>(std::string("vFloat16_nbit_deflate"), dims,1,0,9);
    dset.write(v1);

    // Check that Dataset has been created
    list.clear();
    list = fic.find("vFloat16_nbit_deflate","/","DATASET");
    ASSERT_EQ(list.size(), 1);

    // Read back the values and check that they are correct
    v1r.clear();
    dset.read(v1r);
    ASSERT_EQ(v1r.size(), 1000000);
    ASSERT_EQ(v1r[0], v1[0]);
    ASSERT_EQ(v1r[99], v1[99]);

    // Check the size on disk of the dataset
    long storageSize3 = dset.getStorageSize();
    EXPECT_GT(storageSize2, storageSize3);
}


int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
