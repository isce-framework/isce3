//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <valarray>

#include <isce3/core/LUT2d.h>
#include <isce3/core/Serialization.h>

#include <isce3/io/IH5.h>

TEST(DopplerTest, CheckArchive) {

    // Make an LUT2d for Dopple representation
    isce3::core::LUT2d<double> doppler;

    // Open HDF5 file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing Doppler grid
    isce3::io::IGroup group = file.openGroup(
        "/science/LSAR/SLC/metadata/processingInformation/parameters"
    );

    // Deserialize the Doppler grid
    isce3::core::loadCalGrid(group, "frequencyA/dopplerCentroid", doppler);

    // Deserialize a valarray directly
    std::valarray<double> dopdata;
    isce3::io::loadFromH5(group, "frequencyA/dopplerCentroid", dopdata);
    
    // Check LUT values against valarray values
    for (size_t i = 0; i < doppler.length(); ++i) {
        for (size_t j = 0; j < doppler.width(); ++j) {
            ASSERT_NEAR(doppler.data()(i,j), dopdata[i*doppler.width() + j], 1.0e-10);
        }
    }
}

TEST(DopplerTest, CheckWrite) {

    // Make an LUT2d for Dopple representation
    isce3::core::LUT2d<double> doppler;

    // Load its data
    {
    // Open HDF5 file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing Doppler grid
    isce3::io::IGroup group = file.openGroup(
        "/science/LSAR/SLC/metadata/processingInformation/parameters"
    );

    // Deserialize the Doppler grid
    isce3::core::loadCalGrid(group, "frequencyA/dopplerCentroid", doppler);
    }

    // Write LUT2d to file
    {
    // Create a dummy hdf5 file
    std::string dummyfile("dummy_doppler.h5");
    isce3::io::IH5File dummy(dummyfile, 'x');

    // Write orbit to dataset (use dummy reference epoch)
    const isce3::core::DateTime refEpoch(2020, 1, 1);
    isce3::io::IGroup group = dummy.createGroup("doppler");
    // Need to create sub-group beforehand
    group.createGroup("frequencyA");
    isce3::core::saveCalGrid(group, "frequencyA/dopplerCentroid", doppler, refEpoch, "Hz");
    }

    // Load a new LUT2d
    isce3::core::LUT2d<double> newDoppler;
    std::string h5file("dummy_doppler.h5");
    isce3::io::IH5File file(h5file);
    isce3::io::IGroup group = file.openGroup("doppler");
    isce3::core::loadCalGrid(group, "frequencyA/dopplerCentroid", newDoppler);

    // Check for equality
    ASSERT_EQ(doppler, newDoppler);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
