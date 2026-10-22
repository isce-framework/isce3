import itertools

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from isce3.unwrap.bridge_phase import (bridgeConnectComponent, label_boundary,
                                       label_conn_comp)


@pytest.fixture
def component_mask():
    mask = np.zeros((24, 40), dtype=bool)
    mask[2:10, 2:10] = True       # Survives erosion
    mask[2:10, 16:24] = True     # Survives erosion
    mask[16:18, 30:32] = True    # Disappears during erosion
    return mask


def test_label_conn_comp_erosion(component_mask):
    label_img, num_label = label_conn_comp(
        component_mask,
        min_num_pixel=1,
        erosion_size=3,
    )

    assert num_label == 2
    assert label_img[4, 4] > 0
    assert label_img[4, 18] > 0
    assert label_img[4, 4] != label_img[4, 18]

    # Keep the full original footprint of surviving components.
    expected = component_mask.copy()
    expected[16:18, 30:32] = False
    np.testing.assert_array_equal(label_img > 0, expected)


def test_label_boundary_erosion(component_mask):
    label_img, num_label = label_conn_comp(
        component_mask,
        min_num_pixel=1,
        erosion_size=0,
    )

    label_img, num_label, label_bound = label_boundary(
        label_img,
        num_label,
        erosion_size=1,
    )

    assert num_label == 2

    expected = component_mask.copy()
    expected[16:18, 30:32] = False
    np.testing.assert_array_equal(label_img > 0, expected)

    # Check the exact inner boundary after erosion.
    expected_bound = np.zeros_like(label_img)
    for rows, cols, sample in [
        (slice(3, 9), slice(3, 9), (4, 4)),
        (slice(3, 9), slice(17, 23), (4, 18)),
    ]:
        expected_bound[rows, cols] = label_img[sample]

    expected_bound[4:8, 4:8] = 0
    expected_bound[4:8, 18:22] = 0

    np.testing.assert_array_equal(label_bound, expected_bound)


def test_label_boundary_inconsistent_count(component_mask):
    label_img, _ = label_conn_comp(
        component_mask,
        min_num_pixel=1,
        erosion_size=0,
    )

    # Two components survive, but the supplied count is only one.
    with pytest.raises(
        ValueError,
        match=r"erosion retained 2 unique component labels, "
              r"but num_label is 1",
    ):
        label_boundary(label_img, num_label=1, erosion_size=1)

def test_get_all_bridge_matches_brute_force():
    # Non-convex, nested and edge-touching regions
    conncomp = np.zeros((40, 60), dtype=np.uint32)
    conncomp[2:20, 2:8] = 1
    conncomp[2:8, 2:30] = 1                # L-shape
    conncomp[12:30, 14:40] = 1
    conncomp[16:26, 18:36] = 0             # ring with a hole...
    conncomp[19:23, 24:30] = 1             # ...and an island inside it
    conncomp[30:40, 45:60] = 1             # touches the image edge
    conncomp[0:5, 50:55] = 1

    bridge = bridgeConnectComponent(conncomp)
    bridge.label(min_num_pixel=1, erosion_size=0)
    conn, dist_mat = bridge.get_all_bridge()
    assert bridge.num_label == 5

    # Brute force over every pixel of each region
    points = [np.argwhere(bridge.labelImg == i + 1)
              for i in range(bridge.num_label)]
    for i, j in itertools.combinations(range(bridge.num_label), 2):
        expected = cdist(points[i], points[j]).min()
        assert dist_mat[i, j] == pytest.approx(expected)
        assert dist_mat[j, i] == pytest.approx(expected)

        # Endpoints lie in their regions and realize the distance
        bridge_ij = conn[f"{i + 1}_{j + 1}"]
        yx_i = bridge_ij[str(i + 1)].astype(int)
        yx_j = bridge_ij[str(j + 1)].astype(int)
        assert bridge.labelImg[tuple(yx_i)] == i + 1
        assert bridge.labelImg[tuple(yx_j)] == j + 1
        assert np.hypot(*(yx_i - yx_j)) == pytest.approx(expected)
