//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#include <isce/core/Ellipsoid.h>
#include <isce/core/Serialization.h>

#include <isce/io/IH5.h>


TEST(EllipsoidTest, CheckArchive) {
    // Make an ellipsoid
    isce::core::Ellipsoid ellipsoid;

    // Check validity of product file
    std::string h5file("../../data/envisat.h5");
    ASSERT_TRUE(isce::io::IH5File::isHdf5(h5file));

    // Open the file
    isce::io::IH5File file(h5file);

    // Open group containing ellipsoid
    isce::io::IGroup group = file.openGroup("/science/metadata/identification");

    // Deserialize the ellipsoid
    isce::core::loadFromH5(group, ellipsoid);
    
    // Check values
    ASSERT_NEAR(ellipsoid.a(), 6378137.0, 1.0e-9);
    ASSERT_NEAR(ellipsoid.e2(), 0.0066943799, 1.0e-9);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
