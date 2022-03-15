#include <cmath>

#include <gtest/gtest.h>

#include <isce3/unwrap/snaphu/snaphu.h>

/** Check if two arrays are element-wise equal. */
template<class ArrayLike2D>
bool allEqual(const ArrayLike2D& a, const ArrayLike2D& b)
{
    auto m = a.rows();
    auto n = a.cols();

    // Check that `a` and `b` have the same shape.
    if (b.rows() != m or b.cols() != n) {
        return false;
    }

    // Check that each element of `a` matches the corresponding element from
    // `b`.
    for (decltype(m) i = 0; i < m; ++i) {
        for (decltype(n) j = 0; j < n; ++j) {
            if (a(i, j) != b(i, j)) {
                return false;
            }
        }
    }

    return true;
}

TEST(MCFTest, MCF1)
{
    // Dimensions of the 2D wrapped phase field.
    const long nrow = 3;
    const long ncol = 5;

    // Construct a wrapped phase field with a positive residue in the top-left
    // corner and a negative residue in the bottom-left corner.
    auto wrapped_phase = isce3::unwrap::Array2D<float>(nrow, ncol);
    wrapped_phase.row(0) = M_PI;
    wrapped_phase.row(1) = 0.0f;
    wrapped_phase.row(2) = M_PI;
    wrapped_phase.col(0) = -M_PI;

    std::cout << "wrapped_phase = " << std::endl << wrapped_phase << std::endl;

    // Calculate residues (just for debugging purposes -- not required for
    // testing).
    auto residue = isce3::unwrap::Array2D<signed char>(nrow - 1, ncol - 1);
    isce3::unwrap::CycleResidue(wrapped_phase, residue, nrow, ncol);

    std::cout << "residue = " << std::endl << residue << std::endl;

    // Initialize all arc costs to an arbitrary large value.
    auto costs = isce3::unwrap::MakeRowColArray2D<short>(nrow, ncol);
    costs = 99;

    // Carve out a zero-cost path from the positive residue to the negative
    // residue.
    auto rowcosts = costs.topLeftCorner(nrow - 1, ncol);
    auto colcosts = costs.bottomLeftCorner(nrow, ncol - 1);
    rowcosts(0, 1) = 0;
    rowcosts(0, 2) = 0;
    rowcosts(0, 3) = 0;
    colcosts(1, 3) = 0;
    rowcosts(1, 3) = 0;
    rowcosts(1, 2) = 0;
    rowcosts(1, 1) = 0;

    std::cout << "rowcosts = " << std::endl << rowcosts << std::endl;
    std::cout << "colcosts = " << std::endl << colcosts << std::endl;

    // Calculate arc flows using the MCF initializer.
    isce3::unwrap::Array2D<short> flows;
    isce3::unwrap::MCFInitFlows(wrapped_phase, &flows, costs, nrow, ncol);

    auto rowflows = flows.topLeftCorner(nrow - 1, ncol);
    auto colflows = flows.bottomLeftCorner(nrow, ncol - 1);

    std::cout << "rowflows = " << std::endl << rowflows << std::endl;
    std::cout << "colflows = " << std::endl << colflows << std::endl;

    // Get the expected resulting flows.
    auto true_flows = isce3::unwrap::MakeRowColArray2D<short>(nrow, ncol);
    true_flows = 0;

    auto true_rowflows = true_flows.topLeftCorner(nrow - 1, ncol);
    auto true_colflows = true_flows.bottomLeftCorner(nrow, ncol - 1);
    true_rowflows(0, 1) = 1;
    true_rowflows(0, 2) = 1;
    true_rowflows(0, 3) = 1;
    true_colflows(1, 3) = 1;
    true_rowflows(1, 3) = -1;
    true_rowflows(1, 2) = -1;
    true_rowflows(1, 1) = -1;

    std::cout << "true_rowflows = " << std::endl << true_rowflows << std::endl;
    std::cout << "true_colflows = " << std::endl << true_colflows << std::endl;

    // Make sure the computed flows match the expected flows.
    EXPECT_TRUE(allEqual(rowflows, true_rowflows));
    EXPECT_TRUE(allEqual(colflows, true_colflows));
}

TEST(MCFTest, MCF2)
{
    // Dimensions of the 2D wrapped phase field.
    const long nrow = 8;
    const long ncol = 3;

    // Construct a wrapped phase field with a single positive residue and two
    // negative residues. The residue array looks like this:
    //
    // [ 0  0 ]
    // [-1  0 ]
    // [ 0  0 ]
    // [ 1  0 ]
    // [ 0  0 ]
    // [-1  0 ]
    // [ 0  0 ]
    //
    auto wrapped_phase = isce3::unwrap::Array2D<float>(nrow, ncol);
    wrapped_phase.col(0) = 0.0f;
    wrapped_phase.col(1) = M_PI;
    wrapped_phase.col(2) = M_PI;
    wrapped_phase.row(2) = -M_PI;
    wrapped_phase.row(3) = -M_PI;
    wrapped_phase.row(6) = -M_PI;
    wrapped_phase.row(7) = -M_PI;

    std::cout << "wrapped_phase = " << std::endl << wrapped_phase << std::endl;

    // Calculate residues (just for debugging purposes -- not required for
    // testing).
    auto residue = isce3::unwrap::Array2D<signed char>(nrow - 1, ncol - 1);
    isce3::unwrap::CycleResidue(wrapped_phase, residue, nrow, ncol);

    std::cout << "residue = " << std::endl << residue << std::endl;

    // Initialize all arc costs to an arbitrary large value.
    auto costs = isce3::unwrap::MakeRowColArray2D<short>(nrow, ncol);
    costs = 99;

    // Carve out a zero-cost path between each residue and the "ground" node.
    auto rowcosts = costs.topLeftCorner(nrow - 1, ncol);
    auto colcosts = costs.bottomLeftCorner(nrow, ncol - 1);
    colcosts(0, 0) = 0;
    colcosts(1, 0) = 0;
    rowcosts(3, 1) = 0;
    rowcosts(3, 2) = 0;
    colcosts(6, 0) = 0;
    colcosts(7, 0) = 0;

    std::cout << "rowcosts = " << std::endl << rowcosts << std::endl;
    std::cout << "colcosts = " << std::endl << colcosts << std::endl;

    // Calculate arc flows using the MCF initializer.
    isce3::unwrap::Array2D<short> flows;
    isce3::unwrap::MCFInitFlows(wrapped_phase, &flows, costs, nrow, ncol);

    auto rowflows = flows.topLeftCorner(nrow - 1, ncol);
    auto colflows = flows.bottomLeftCorner(nrow, ncol - 1);

    std::cout << "rowflows = " << std::endl << rowflows << std::endl;
    std::cout << "colflows = " << std::endl << colflows << std::endl;

    // Get the expected resulting flows.
    auto true_flows = isce3::unwrap::MakeRowColArray2D<short>(nrow, ncol);
    true_flows = 0;

    auto true_rowflows = true_flows.topLeftCorner(nrow - 1, ncol);
    auto true_colflows = true_flows.bottomLeftCorner(nrow, ncol - 1);
    true_colflows(0, 0) = 1;
    true_colflows(1, 0) = 1;
    true_rowflows(3, 1) = 1;
    true_rowflows(3, 2) = 1;
    true_colflows(6, 0) = -1;
    true_colflows(7, 0) = -1;

    std::cout << "true_rowflows = " << std::endl << true_rowflows << std::endl;
    std::cout << "true_colflows = " << std::endl << true_colflows << std::endl;

    // Make sure the computed flows match the expected flows.
    EXPECT_TRUE(allEqual(rowflows, true_rowflows));
    EXPECT_TRUE(allEqual(colflows, true_colflows));
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
