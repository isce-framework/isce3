
import itertools
import journal
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.ndimage import (binary_erosion,
                           find_objects,
                           binary_dilation)
from scipy.sparse import csgraph as csg
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any, List


def bridge_unwrapped_phase(unw_phase: np.ndarray,
                           radius: int,
                           min_num_pixel: int,
                           erosion_size: int,
                           ramp_type: str,
                           deramp_max_num_sample: int) -> np.ndarray:
    """Bridge disconnected unwrapped phase regions by estimating and removing
    their relative phase offsets, which is interger of 2 pi.

    In unwrapping, isolated components may be unwrapped independently,
    producing integer cycle jumps between them. This function finds
    nearby components (within `radius` pixels), computes the median phase
    difference at their nearest boundary pixels, shifts one region to align it,
    and merges the labels—effectively “bridging” the jump.
    Finally, it filters out speckle (small islands) and smooths
    mask edges via erosion.

    Parameters
    ----------
    unw_phase : numpy.ndarray
        2D array representing the unwrapped phase image.
    radius : int
        The maximum radius used when bridging disconnected regions.
    min_num_pixel : int
        The minimum number of pixel of connected components to retain during processing.
    erosion_size : int
        The size of the structuring element used for erosion during labeling.
    ramp_type : str, optional
        Type of ramp to be estimated. Default is 'linear'.
        Possible options -
        'linear', 'quadratic', 'linear_range', 'linear_azimuth'
        'quadratic_range', 'quadratic_azimuth'
    deramp_max_num_sample : float, optional
        Maximum number of pixel samples, above which uniform sampling is
        applied to reduce sample size. Default is 1e6.    Returns
    -------
    bridge_unw : numpy.ndarray
        2D array with the bridged unwrapped phase image.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.bridge_unwrapped_phase")

    runw_img_bool = unw_phase != 0
    label_img, num_cluster = nd_label(runw_img_bool, structure=np.ones((3, 3)))
    channel.log(f"Bridge algorithm : {num_cluster} clusters")

    if num_cluster <= 1:
        channel.log(f"Bridge algorithm is not applied since all components are connected.")
        return unw_phase

    channel.log(f"   radius: {radius}   min_num_pixel: {min_num_pixel}  erosion_size: {erosion_size} ")
    cc = bridgeConnectComponent(conncomp=label_img)
    cc.label(min_num_pixel=min_num_pixel, erosion_size=erosion_size)
    if cc.label_ref is not None:
        cc.find_mst_bridge()
        bridge_unw = cc.unwrap_conn_comp(
            unw_phase,
            radius=radius,
            ramp_type=ramp_type,
            max_num_sample=deramp_max_num_sample
            )
    else:
        bridge_unw = unw_phase
    return bridge_unw


def label_boundary(
        label_img: np.ndarray,
        num_label: int,
        erosion_size: int = 5,
        ) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Label the boundary of the labeled array.

    This function labels the boundaries of connected components in a labeled
    array. It optionally applies morphological erosion to the labeled array
    before finding and labeling the boundaries.

    Parameters
    ----------
    label_img : numpy.ndarray
        2D array of labeled regions.
    num_label : int
        The number of labels in the labeled array.
    erosion_size : int, optional
        The size of the structuring element used for morphological erosion.
        Default is 5.

    Returns
    -------
    label_img : numpy.ndarray
        2D array with relabeled regions after erosion and boundary labeling.
    num_label : int
        The updated number of labels after relabeling.
    label_bound : numpy.ndarray
        2D array of labeled boundaries of the regions.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.label_boundary")

    if erosion_size > 0:
        erosion_yy, erosion_xx = np.ogrid[-erosion_size:erosion_size + 1,
                                          -erosion_size:erosion_size + 1]
        # circle mask
        erosion_structure = (erosion_xx ** 2 +
                             erosion_yy ** 2) <= (erosion_size**2)
        label_erosion_img = binary_erosion(
            label_img,
            structure=erosion_structure).astype(np.uint8)

        labeled_array, _ = nd_label(label_erosion_img)
        regions = find_objects(labeled_array)

        if len(regions) < num_label:
            channel.log(
                "Regions lost during morphological erosion operation:")
            erosion_labels = [label_erosion_img[region].max()
                              for region in regions]
            for i in range(1, num_label + 1):
                if i not in erosion_labels:
                    label_img[label_img == i] = 0

    else:
        label_erosion_img = label_img > 0
    label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))
    # Create a boundary map using binary dilation and subtracting the original image
    boundary_img = binary_dilation(label_erosion_img) & ~label_erosion_img
    label_bound = boundary_img.astype(np.uint8)
    label_bound *= label_erosion_img

    return label_img, num_label, label_bound


def label_conn_comp(
        mask: np.ndarray,
        min_num_pixel: float = 2.5e3,
        erosion_size: int = 5,
        ) -> Tuple[np.ndarray, int]:
    """
    Label and clean up the connected components mask.

    This function labels the connected components in a binary mask,
    removes small objects below a specified area, and optionally applies
    morphological erosion to refine the labels.

    Parameters
    ----------
    mask : numpy.ndarray
        2D binary array representing the mask of connected components.
    min_num_pixel : float, optional
        Minimum number of pixels for a region to be kept. Default is 2.5e3.
    erosion_size : int, optional
        Size of the structuring element used for morphological erosion.
        Default is 5.

    Returns
    -------
    label_img : numpy.ndarray
        2D array with labeled connected components after cleaning and erosion.
    num_label : int
        The number of labels (connected components) after cleaning and erosion.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.label_conn_comp")

    # Label the connected components
    label_img, num_label = nd_label(mask, structure=np.ones((3, 3)))

    # Calculate min_num_pixel if not specified
    channel.log(f"Removing regions with area < {int(min_num_pixel)}")

    # Remove small objects
    object_slices = find_objects(label_img)
    for i, slice_ in enumerate(object_slices, start=1):
        if slice_ is not None:
            if np.sum(label_img[slice_] == i) < min_num_pixel:
                label_img[label_img == i] = 0

    # Re-label after removing small objects
    label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))

    # Apply morphological erosion if specified
    if erosion_size > 0:
        erosion_structure = np.ones((erosion_size, erosion_size), dtype=bool)
        label_erosion_img = binary_erosion(
            label_img > 0,
            structure=erosion_structure).astype(np.uint8)

        labeled_array, _ = nd_label(label_erosion_img)
        regions = find_objects(labeled_array)

        if len(regions) < num_label:
            channel.log("Regions lost during morphological erosion operation:")
            erosion_labels = [label_erosion_img[region].max()
                              for region in regions]
            for i in range(1, num_label + 1):
                if i not in erosion_labels:
                    label_img[label_img == i] = 0

        # Re-label after erosion
        label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))

    return label_img, num_label


class bridgeConnectComponent:
    """Object for bridging connected components."""

    def __init__(self, conncomp: np.ndarray):
        """Initialize the ConnectComponent object."""
        if not isinstance(conncomp, np.ndarray):
            raise ValueError("Input conncomp is not np.ndarray")
        self.conncomp = conncomp
        self.length, self.width = self.conncomp.shape

    def label(self,
              min_num_pixel: float = 2.5e3,
              erosion_size: int = 5) -> None:
        """
        Label connected components in the image and identify the reference
        label.

        This function labels connected components in the input image based on a
        minimum area threshold and performs boundary erosion. It also
        identifies the reference label as the largest connected component.

        Parameters
        ----------
        min_num_pixel : float, optional
            Minimum area threshold for connected components. Default is 2500.
        erosion_size : int, optional
            Size of the structuring element for boundary erosion. Default is 5.

        Returns
        -------
        None
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent")
        self.labelImg, self.num_label = label_conn_comp(
            self.conncomp, min_num_pixel=min_num_pixel)

        if self.num_label == 1:
            channel.log(f"Bridge algorithm is not applied because only one component exists.")
        elif self.num_label == 0:
            channel.log(f"Bridge algorithm is not applied because component does not exist.")

        self.labelImg, self.num_label, self.labelBound = label_boundary(
            self.labelImg,
            self.num_label,
            erosion_size=erosion_size
            )

        regions = find_objects(self.labelImg)
        # if regions are not empty
        if regions:
            idx = np.argmax([np.sum(self.labelImg[region])
                             for region in regions])
            self.label_ref = idx + 1
        # if regions are empty
        else:
            self.label_ref = None

    def get_all_bridge(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Compute all possible bridges between labeled areas and their distances.

        This function calculates the shortest distances between all pairs of
        labeled areas in the label image using k-d trees for efficient
        nearest-neighbor searches. It stores the connections and distances in
        a dictionary and a distance matrix.

        Parameters
        ----------
        None

        Returns
        -------
        connDict : Dict[str, Any]
            A dictionary containing the coordinates of the closest points
            between each pair of labeled areas and their distances. Keys
            are formatted as "{label1}_{label2}" with values as dictionaries
            containing:
            - label1: coordinates of the closest point in the first labeled
                      area.
            - label2: coordinates of the closest point in the second labeled
                      area.
            - "distance": the distance between these points.
        distMat : np.ndarray
            A symmetric matrix of shape (num_label, num_label) containing the
            minimum distances between each pair of labeled areas.
        """
        self.connDict = {}
        self.distMat = np.zeros((self.num_label, self.num_label),
                                dtype=np.float32)
        # if the number of labels is zero, then return empty dictionary and
        # zero array
        if self.num_label == 0:
            return self.connDict, self.distMat

        trees = [cKDTree(np.argwhere(self.labelImg == i + 1))
                 for i in range(self.num_label)]

        for i, j in itertools.combinations(range(self.num_label), 2):
            dist, idx = trees[i].query(trees[j].data)
            idx_min = np.argmin(dist)
            yxi = trees[i].data[idx[idx_min]]
            yxj = trees[j].data[idx_min]
            dist_min = dist[idx_min]
            n0, n1 = str(i + 1), str(j + 1)
            conn = {n0: yxi, n1: yxj, "distance": dist_min}
            self.connDict[f"{n0}_{n1}"] = conn
            self.distMat[i, j] = self.distMat[j, i] = dist_min

        return self.connDict, self.distMat

    def find_mst_bridge(self) -> List[Dict[str, Union[int, float]]]:
        """
        Search for bridges to connect all labeled areas using the minimum
        spanning tree algorithm.

        This function finds the minimum set of connections (bridges) needed to
        connect all labeled areas using the minimum spanning tree (MST)
        algorithm. If the distance matrix (`self.distMat`) is not
        available, it is computed using the `get_all_bridge` method.

        Parameters
        ----------
        None

        Returns
        -------
        bridges : List[Dict[str, Union[int, float]]]
            A list of dictionaries, each representing a bridge with the
            following keys:
            - 'first_endpoint_x': x-coordinate of the first node.
            - 'first_endpoint_y': y-coordinate of the first node.
            - 'second_endpoint_x': x-coordinate of the second node.
            - 'second_endpoint_y': y-coordinate of the second node.
            - 'label0': label of the first node.
            - 'label1': label of the second node.
            - 'distance': Euclidean distance between the two nodes.
        """
        if not hasattr(self, "distMat"):
            self.get_all_bridge()

        distMatMst = csg.minimum_spanning_tree(self.distMat)
        succs, preds = csg.breadth_first_order(
            distMatMst, i_start=self.label_ref - 1, directed=False
        )
        self.bridges = []
        for i in range(1, succs.size):
            n0 = preds[succs[i]] + 1
            n1 = succs[i] + 1
            if n0 > n1:
                nn = [str(n1), str(n0)]
            else:
                nn = [str(n0), str(n1)]
            conn = self.connDict[f"{nn[0]}_{nn[1]}"]
            first_endpoint_y, first_endpoint_x = conn[str(n0)]
            second_endpoint_y, second_endpoint_x = conn[str(n1)]
            bridge = dict()
            bridge = {
                "first_endpoint_x": int(first_endpoint_x),
                "first_endpoint_y": int(first_endpoint_y),
                "second_endpoint_x": int(second_endpoint_x),
                "second_endpoint_y": int(second_endpoint_y),
                "distance": float(
                    ((second_endpoint_x - first_endpoint_x)**2 +
                     (second_endpoint_y - first_endpoint_y)**2)**0.5),
                "label0": n0,
                "label1": n1,
            }
            self.bridges.append(bridge)
        self.num_bridge = len(self.bridges)
        return self.bridges

    def get_bridge_endpoint_aoi_mask(
        self, bridge: Dict[str, int], radius: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Area of Interest (AOI) mask for bridge endpoints.

        This function generates AOI masks around the endpoints of a bridge
        between two connected components in the phase data. The AOI masks are
        square regions centered at the bridge endpoints with a specified
        radius.

        Parameters
        ----------
        bridge : Dict[str, int]
            Dictionary containing the coordinates of the bridge endpoints.
            Keys should include:
            - "first_endpoint_x": x-coordinate of the first endpoint.
            - "first_endpoint_y": y-coordinate of the first endpoint.
            - "second_endpoint_x": x-coordinate of the second endpoint.
            - "second_endpoint_y": y-coordinate of the second endpoint.
        radius : int, optional
            Radius of the square AOI around each bridge endpoint.
            Default is 50.

        Returns
        -------
        aoi_mask0 : np.ndarray
            Boolean mask of the AOI around the first bridge endpoint.
        aoi_mask1 : np.ndarray
            Boolean mask of the AOI around the second bridge endpoint.
        """
        # 1) unpack the two endpoints
        first_endpoint_x = bridge["first_endpoint_x"]
        first_endpoint_y = bridge["first_endpoint_y"]
        second_endpoint_x = bridge["second_endpoint_x"]
        second_endpoint_y = bridge["second_endpoint_y"]

        # 2) compute row/col bounds for the first endpoint
        first_col_min = max(0, first_endpoint_x - radius)
        first_col_max = min(self.width, first_endpoint_x  + radius + 1)
        first_row_min = max(0, first_endpoint_y - radius)
        first_row_max = min(self.length, first_endpoint_y  + radius + 1)

        # 3) compute row/col bounds for the second endpoint
        second_col_min = max(0, second_endpoint_x - radius)
        second_col_max = min(self.width, second_endpoint_x + radius + 1)
        second_row_min = max(0, second_endpoint_y - radius)
        second_row_max = min(self.length, second_endpoint_y + radius + 1)

        # 4) build the two AOI masks
        aoi_mask_first = np.zeros(self.labelImg.shape, dtype=bool)
        aoi_mask_second = np.zeros(self.labelImg.shape, dtype=bool)

        aoi_mask_first[
            first_row_min:first_row_max,
            first_col_min:first_col_max
        ] = True

        aoi_mask_second[
            second_row_min:second_row_max,
            second_col_min:second_col_max
        ] = True

        return aoi_mask_first, aoi_mask_second

    def unwrap_conn_comp(
        self,
        unw: np.ndarray,
        radius: int = 50,
        ramp_type: Optional[str] = None,
        max_num_sample: int = 1e6,
    ) -> np.ndarray:
        """
        Perform bridging to unwrap connected components in the phase data.

        This function unrolls the phase data by bridging the gaps between
        connected components, optionally removing any ramp present in the data.

        Parameters
        ----------
        unw : np.ndarray
            2D array of unwrapped phase data.
        radius : int, optional
            Radius of the area of interest (AOI) around bridge endpoints for
            unwrapping. Default is 50.
        ramp_type : Optional[str], optional
            Type of ramp to be estimated and removed before unwrapping.
            If None, no ramp is removed. Default is None.

        Returns
        -------
        unw : np.ndarray
            2D array of unwrapped phase data after bridging.
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent")
        radius = int(min(radius, min(self.conncomp.shape) * 0.5))
        unw = np.array(unw, dtype=np.float32)

        if ramp_type is not None:
            channel.log(f"Estimating a {ramp_type} ramp")
            ramp_mask = self.labelImg == self.label_ref
            unw, ramp = deramp(unw, ramp_mask, ramp_type, max_num_sample)

        for bridge in self.bridges:
            aoi_mask0, aoi_mask1 = self.get_bridge_endpoint_aoi_mask(
                bridge, radius=radius
            )
            label_mask0 = self.labelImg == bridge["label0"]
            label_mask1 = self.labelImg == bridge["label1"]
            value0 = np.nanmedian(unw[np.logical_and(aoi_mask0, label_mask0)])
            value1 = np.nanmedian(unw[np.logical_and(aoi_mask1, label_mask1)])
            diff_value = value1 - value0
            num_jump = (np.abs(diff_value) + np.pi) // (2.0 * np.pi)
            if diff_value > 0:
                num_jump *= -1
            unw[label_mask1] += 2.0 * np.pi * num_jump

        if ramp_type is not None:
            unw += ramp

        return unw


def deramp(data,
           mask_in=None,
           ramp_type='linear',
           max_num_sample=1e6,
           ignore_zero_value=True):
    """Remove ramp from input data matrix based on pixel marked by
    mask. Ignore data with NaN or zero value.

    Parameters
    ----------
    data : np.ndarray
        2D or 3D array of data to be deramped. If 3D, it's in the size of
        (num_date, length, width).
    mask_in : np.ndarray, optional
        2D array mask of pixels used for ramp estimation.
    ramp_type : str, optional
        Type of ramp to be estimated. Default is 'linear'.
        Possible options -
        'linear', 'quadratic', 'linear_range', 'linear_azimuth'
        'quadratic_range', 'quadratic_azimuth'
    max_num_sample : float, optional
        Maximum number of pixel samples, above which uniform sampling is
        applied to reduce sample size. Default is 1e6.
    ignore_zero_value : bool, optional
        Ignore pixels with zero values. Default is True.
        Recommended: True for phase data and False for offset data.

    Returns
    -------
    data_out : np.ndarray
        2D or 3D array of data after deramping.
    ramp : np.ndarray
        2D or 3D array of the estimated ramp.
    """
    dshape = data.shape
    length, width = dshape[-2:]
    num_pixel = length * width

    # prepare input data
    if len(dshape) == 3:
        data = np.moveaxis(data, 0, -1)
        data = data.reshape(num_pixel, -1)
        dmean = np.mean(data, axis=-1).flatten()
    else:
        data = data.reshape(-1, 1)
        dmean = np.array(data).flatten()

    # mask
    # 1. default
    if mask_in is None:
        mask_in = np.ones((length, width), dtype=np.float32)
    mask = (mask_in != 0).flatten()
    del mask_in

    # 2. ignore pixels with NaN and/or zero data value
    mask *= ~np.isnan(dmean)
    if ignore_zero_value:
        mask *= dmean != 0.
    del dmean

    # 3. for big dataset: uniformally sample the data for ramp estimation
    mask_sum = np.sum(mask) 
    if max_num_sample and mask_sum > max_num_sample:
        step = int(np.ceil(np.sqrt(mask_sum / max_num_sample)))
        if step > 1:
            sample_flag = np.zeros((length, width), dtype=np.bool_)
            sample_flag[int(step/2)::step,
                        int(step/2)::step] = 1
            mask *= sample_flag.flatten()
            del sample_flag

    # design matrix
    xx, yy = np.meshgrid(np.arange(0, width),
                         np.arange(0, length))
    xx = np.array(xx, dtype=np.float32).reshape(-1, 1)
    yy = np.array(yy, dtype=np.float32).reshape(-1, 1)
    ones = np.ones(xx.shape, dtype=np.float32)
    if ramp_type == 'linear':
        G = np.hstack((yy, xx, ones))
    elif ramp_type == 'quadratic':
        G = np.hstack((yy**2, xx**2, yy*xx, yy, xx, ones))
    elif ramp_type == 'linear_range':
        G = np.hstack((xx, ones))
    elif ramp_type == 'linear_azimuth':
        G = np.hstack((yy, ones))
    elif ramp_type == 'quadratic_range':
        G = np.hstack((xx**2, xx, ones))
    elif ramp_type == 'quadratic_azimuth':
        G = np.hstack((yy**2, yy, ones))
    else:
        raise ValueError(f'un-recognized ramp type: {ramp_type}')

    # estimate ramp
    X = np.dot(np.linalg.pinv(G[mask, :], rcond=1e-15), data[mask, :])
    ramp = np.dot(G, X)
    ramp = np.array(ramp, dtype=data.dtype)

    # do not change pixel with original zero value
    if ignore_zero_value:
        ramp[data == 0] = 0

    data_out = data - ramp
    if len(dshape) == 3:
        ramp = np.moveaxis(ramp, -1, 0)
        data_out = np.moveaxis(data_out, -1, 0)
    ramp = ramp.reshape(dshape)
    data_out = data_out.reshape(dshape)
    return data_out, ramp
