
import numpy as np
from PIL import Image


def find_v_junction(mask, min_gap=20, min_height=40, min_width=3):
    """
    Finds the V junction by detecting two branches
    separated by a substantial gap.
    
    Returns (x, y) or None.
    """

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    height, width = mask.shape

    rows, cols = np.where(mask)

    if len(rows) == 0:
        return None

    top = rows.min()
    bottom = rows.max()

    # Search for the V in the lower 85% of the tree
    search_top = int(top + 0.15 * (bottom - top))

    gaps = []

    for y in range(search_top, bottom + 1):

        row = mask[y]

        # Find continuous white sections
        changes = np.diff(
            np.concatenate(([False], row, [False])).astype(int)
        )

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        widths = ends - starts

        # Ignore very small sections
        valid = widths >= min_width

        starts = starts[valid]
        ends = ends[valid]

        if len(starts) < 2:
            gaps.append(None)
            continue

        # Find the two largest sections
        indices = np.argsort(ends - starts)[-2:]

        sections = sorted(
            [(starts[i], ends[i]) for i in indices]
        )

        left, right = sections

        gap_width = right[0] - left[1]

        if gap_width >= min_gap:

            gap_centre = (left[1] + right[0]) / 2

            gaps.append((gap_centre, y))

        else:
            gaps.append(None)

    # Find continuous runs of rows containing a gap
    runs = []
    current_run = []

    for point in gaps:

        if point is not None:
            current_run.append(point)

        else:
            if len(current_run) >= min_height:
                runs.append(current_run)

            current_run = []

    if len(current_run) >= min_height:
        runs.append(current_run)

    if not runs:
        return None

    # Choose the run extending furthest down the tree
    best_run = max(runs, key=lambda run: run[-1][1])

    # Bottom of the V-shaped gap
    x, y = best_run[-1]

    return int(x), int(y)


def scale_gt(gt, target_shape, scale=0.75):
    """
    Resize the GT relative to the reconstructed image.
    
    target_shape: (height, width)
    scale: additional scaling factor
    """

    gt = np.asarray(gt)

    # Convert GT to binary
    if gt.ndim == 3:
        gt = np.any(gt > 0, axis=2)
    else:
        gt = gt > 0

    height, width = target_shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    gt_scaled = np.array(
        Image.fromarray(gt.astype(np.uint8)).resize(
            (new_width, new_height),
            Image.Resampling.NEAREST
        )
    ).astype(bool)

    return gt_scaled


def align_gt(gt_scaled, target_shape, gt_junction, target_junction):
    """
    Translate the scaled GT so that the two V junctions align.
    
    Returns a binary matrix with the target dimensions.
    """

    if gt_junction is None or target_junction is None:
        raise ValueError("Cannot align: V junction not found.")

    height, width = target_shape[:2]

    gt_scaled = np.asarray(gt_scaled).astype(bool)

    # Calculate translation
    x_offset = target_junction[0] - gt_junction[0]
    y_offset = target_junction[1] - gt_junction[1]

    # Create empty matrix
    gt_aligned = np.zeros((height, width), dtype=bool)

    gt_height, gt_width = gt_scaled.shape

    # Destination boundaries
    x1 = max(0, x_offset)
    y1 = max(0, y_offset)

    x2 = min(width, x_offset + gt_width)
    y2 = min(height, y_offset + gt_height)

    # Source boundaries
    src_x1 = x1 - x_offset
    src_y1 = y1 - y_offset

    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    # Copy GT into aligned matrix
    if x2 > x1 and y2 > y1:
        gt_aligned[y1:y2, x1:x2] = gt_scaled[
            src_y1:src_y2,
            src_x1:src_x2
        ]

    return gt_aligned


def align_tree(prediction, gt, scale=0.75):
    """
    Complete tree alignment process.

    1. Scale ground truth
    2. Find both V junctions
    3. Translate GT to align junctions

    Returns:
        gt_aligned
        predicted_junction
        gt_junction
    """

    prediction = np.asarray(prediction).astype(bool)

    # Scale GT
    gt_scaled = scale_gt(gt, prediction.shape, scale)

    # Find V junctions
    predicted_junction = find_v_junction(
        prediction,
        min_gap=20,
        min_height=40
    )

    gt_junction = find_v_junction(
        gt_scaled,
        min_gap=15,
        min_height=25
    )

    # Align GT
    gt_aligned = align_gt(
        gt_scaled,
        prediction.shape,
        gt_junction,
        predicted_junction
    )

    return gt_aligned, predicted_junction, gt_junction
