//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#include <isce/core/Metadata.h>
#include <isce/core/Serialization.h>

#include <isce/io/IH5.h>


TEST(MetadataTest, CheckArchive) {

    // Make a metadata object
    isce::core::Metadata meta;

    // Open product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Deserialize the metadata
    isce::core::load(file, meta, "primary");

    // Check values
    ASSERT_NEAR(meta.slantRangePixelSpacing, 7.803973670948287, 1.0e-10);
    ASSERT_NEAR(meta.rangeFirstSample, 826988.6900674499, 1.0e-10);
    ASSERT_EQ(meta.sensingStart.isoformat(), "2003-02-26T17:55:30.843491759");
    ASSERT_NEAR(meta.prf, 1652.415691672402, 1.0e-10);
    ASSERT_NEAR(meta.radarWavelength, 0.05623564240544047, 1.0e-10);
    ASSERT_EQ(meta.lookSide, -1);

}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
