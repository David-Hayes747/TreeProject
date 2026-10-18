
"""
Alignment of summer ground-truth masks with winter tree segmentations.

This module provides functions for estimating the V-junction of a tree
and aligning its summer ground-truth mask with a predicted winter tree
segmentation.

Alignment involves three stages:
    1. Resizing the summer ground-truth mask.
    2. Identifying V-shaped gaps between tree branches.
    3. Translating the ground truth to align the detected junctions.

The alignment is approximate because the images may differ in
camera viewpoint, scale, season and visible branch structure.

The resulting aligned masks are intended for approximate quantitative
comparison rather than exact pixel-level validation.
"""

import numpy as np
from PIL import Image


def find_v_junction(mask, min_gap=20, min_height=40, min_width=3):
    """
    Estimate the V-junction position in a binary tree mask.

    Searches horizontal image rows for two sufficiently wide foreground
    regions separated by a gap. Consecutive rows containing such a gap
    are grouped, and the lowest qualifying run is selected.

    Args:
        mask (np.ndarray): Binary or nonzero-valued image mask.
        min_gap (int, optional): Minimum horizontal gap between regions.
            Defaults to 20.
        min_height (int, optional): Minimum number of consecutive rows
            containing a qualifying gap. Defaults to 40.
        min_width (int, optional): Minimum width of each foreground
            region. Defaults to 3.

    Returns:
        tuple[int, int] or None: Estimated junction coordinates (x, y),
            or None if no qualifying gap is detected.

    Notes:
        This method is heuristic and may identify background gaps
        rather than the true branch junction.

        The search excludes the upper 15% of the detected tree's
        vertical extent. Detection depends on the selected thresholds
        and the tree's visible structure.
    """

    # Convert the input into a two-dimensional Boolean mask.
    # For multi-channel images, a pixel is foreground if any channel
    # contains a nonzero value.
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    # Find the vertical extent of the detected foreground.
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return None

    top = rows.min()
    bottom = rows.max()

    # Exclude the upper 15% of the tree's vertical extent.
    # The remaining region is searched for a persistent V-shaped gap.
    search_top = int(top + 0.15 * (bottom - top))

    gaps = []

    for y in range(search_top, bottom + 1):

        row = mask[y]

        # Detect transitions between background and foreground.
        # These transitions define continuous horizontal regions.
        changes = np.diff(
            np.concatenate(([False], row, [False])).astype(int)
        )

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        widths = ends - starts

        # Ignore narrow regions that are unlikely to represent
        # substantial branches.
        valid = widths >= min_width

        starts = starts[valid]
        ends = ends[valid]

        if len(starts) < 2:
            gaps.append(None)
            continue

        # Select the two widest foreground regions in this row.
        indices = np.argsort(ends - starts)[-2:]

        # Sort the selected regions from left to right.
        sections = sorted(
            [(starts[i], ends[i]) for i in indices]
        )

        left, right = sections

        # Measure the horizontal separation between the regions.
        gap_width = right[0] - left[1]

        if gap_width >= min_gap:

            gap_centre = (left[1] + right[0]) / 2

            gaps.append((gap_centre, y))

        else:
            gaps.append(None)

    # --------------------------------------------------
    # FIND CONTINUOUS GAP REGIONS
    # --------------------------------------------------

    # A V-junction candidate must persist across several consecutive
    # image rows rather than appearing in only one row.
    runs = []
    current_run = []

    for point in gaps:

        if point is not None:
            current_run.append(point)

        else:
            if len(current_run) >= min_height:
                runs.append(current_run)

            current_run = []

    # Include a qualifying run that continues to the final row.
    if len(current_run) >= min_height:
        runs.append(current_run)

    if not runs:
        return None

    # Select the qualifying run extending furthest down the image.
    best_run = max(runs, key=lambda run: run[-1][1])

    # Use the lowest point of the selected gap as the estimated
    # junction position.
    x, y = best_run[-1]

    return int(x), int(y)


def scale_gt(gt, target_shape, scale=0.75):
    """
    Resize the summer ground-truth mask for comparison with a prediction.

    Converts the ground truth into a Boolean mask and resizes it using
    nearest-neighbour interpolation to preserve binary pixel values.

    Args:
        gt (np.ndarray): Ground-truth image or binary mask.
        target_shape (tuple): Target image dimensions (height, width).
        scale (float, optional): Fraction of the target image dimensions
            used for resizing the ground truth. Defaults to 0.75.

    Returns:
        np.ndarray: Resized two-dimensional Boolean ground-truth mask.

    Notes:
        The scaling factor is empirically selected rather than
        determined through camera calibration.

        The original ground-truth aspect ratio is not explicitly
        preserved because the new dimensions are calculated from
        the target image dimensions.
    """

    gt = np.asarray(gt)

    # Convert the ground truth into a binary mask.
    if gt.ndim == 3:
        gt = np.any(gt > 0, axis=2)
    else:
        gt = gt > 0

    height, width = target_shape[:2]

    # Calculate the resized dimensions relative to the target image.
    # Ensure neither dimension is smaller than one pixel.
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    # Nearest-neighbour interpolation prevents intermediate values
    # from being introduced when resizing the binary mask.
    gt_scaled = np.array(
        Image.fromarray(gt.astype(np.uint8)).resize(
            (new_width, new_height),
            Image.Resampling.NEAREST
        )
    ).astype(bool)

    return gt_scaled


def align_gt(gt_scaled, target_shape, gt_junction, target_junction):
    """
    Translate a scaled ground-truth mask to align two V-junctions.

    Calculates the horizontal and vertical displacement between the
    detected junctions and copies the translated ground truth into
    a mask matching the target image dimensions.

    Args:
        gt_scaled (np.ndarray): Scaled binary ground-truth mask.
        target_shape (tuple): Target image dimensions (height, width).
        gt_junction (tuple[int, int]): Junction coordinates in the
            scaled ground truth.
        target_junction (tuple[int, int]): Junction coordinates in
            the predicted tree mask.

    Returns:
        np.ndarray: Boolean ground-truth mask aligned to the target
            image dimensions. Pixels outside the translated mask
            are set to False.

    Raises:
        ValueError: If either V-junction is missing.

    Notes:
        This method applies translation only. It does not correct
        differences in perspective, rotation or tree structure.

        Parts of the ground truth falling outside the target image
        boundaries are cropped.
    """

    # Both junctions are required to calculate the translation.
    if gt_junction is None or target_junction is None:
        raise ValueError("Cannot align: V junction not found.")

    height, width = target_shape[:2]

    gt_scaled = np.asarray(gt_scaled).astype(bool)

    # Calculate the displacement required to align the junctions.
    x_offset = target_junction[0] - gt_junction[0]
    y_offset = target_junction[1] - gt_junction[1]

    # Initialise the aligned ground-truth mask.
    gt_aligned = np.zeros((height, width), dtype=bool)

    gt_height, gt_width = gt_scaled.shape

    # Calculate destination boundaries within the target image.
    # Clipping prevents the translated mask from exceeding the image.
    x1 = max(0, x_offset)
    y1 = max(0, y_offset)

    x2 = min(width, x_offset + gt_width)
    y2 = min(height, y_offset + gt_height)

    # Calculate corresponding source boundaries in the scaled mask.
    src_x1 = x1 - x_offset
    src_y1 = y1 - y_offset

    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    # Copy the overlapping region into the target-sized mask.
    if x2 > x1 and y2 > y1:
        gt_aligned[y1:y2, x1:x2] = gt_scaled[
            src_y1:src_y2,
            src_x1:src_x2
        ]

    return gt_aligned


def align_tree(prediction, gt, scale=0.75):
    """
    Align summer ground truth with a predicted winter tree mask.

    Performs the complete alignment process:
        1. Resize the summer ground-truth mask.
        2. Estimate V-junctions in both masks.
        3. Translate the ground truth to align the junctions.

    Args:
        prediction (np.ndarray): Predicted winter tree segmentation mask.
        gt (np.ndarray): Summer ground-truth image or binary mask.
        scale (float, optional): Empirical scaling factor applied to
            the ground truth. Defaults to 0.75.

    Returns:
        tuple: Contains:
            - gt_aligned (np.ndarray): Aligned Boolean ground-truth mask.
            - predicted_junction (tuple[int, int]): Junction coordinates
              in the predicted mask.
            - gt_junction (tuple[int, int]): Junction coordinates
              in the scaled ground truth.

    Raises:
        ValueError: If a qualifying V-junction cannot be detected
            in either mask.

    Notes:
        Different V-junction detection thresholds are used for the
        prediction and ground truth to account for differences
        in their appearance.

        Alignment is approximate because the summer and winter
        images may differ in viewpoint, scale and visible structure.

        Evaluation metrics calculated from the aligned masks should
        therefore be interpreted with these limitations in mind.
    """

    # Convert the predicted segmentation into a Boolean mask.
    prediction = np.asarray(prediction).astype(bool)

    # Resize the summer ground truth relative to the winter prediction.
    gt_scaled = scale_gt(gt, prediction.shape, scale)

    # Estimate the V-junction in the predicted winter tree.
    predicted_junction = find_v_junction(
        prediction,
        min_gap=20,
        min_height=40
    )

    # Estimate the V-junction in the scaled summer ground truth.
    # Different thresholds account for differences in image structure.
    gt_junction = find_v_junction(
        gt_scaled,
        min_gap=15,
        min_height=25
    )

    # Translate the ground truth so the estimated junctions coincide.
    # align_gt raises ValueError if either junction was not detected.
    gt_aligned = align_gt(
        gt_scaled,
        prediction.shape,
        gt_junction,
        predicted_junction
    )

    return gt_aligned, predicted_junction, gt_junction
