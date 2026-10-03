import numpy as np
import pytest

from isce3.unwrap.bridge_phase import label_boundary, label_conn_comp


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