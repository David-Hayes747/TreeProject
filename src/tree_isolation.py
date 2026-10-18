
"""
Tree isolation using RGB-D image segmentation.

This module isolates an individual apple tree from its winter RGB-D
observations using depth thresholding, colour segmentation,
morphological dilation, ground removal and connected-component analysis.

The output is a binary mask identifying pixels estimated to belong
to the tree.

The method assumes that the tree forms the largest connected
foreground region after ground removal.

Limitations:
    - Depth values are treated as encoded pixel intensities rather
      than calibrated physical distances.
    - Colour and depth thresholds are empirically selected.
    - Ground removal assumes the ground occupies broad image rows.
    - The largest connected region may not always represent the tree.
    - Thin or disconnected branches may be excluded.
"""

# Import libraries
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation
from .dataset import TreeDataset

# Locate the project root relative to this source file.
# This allows the dataset path to be independent of the working directory.
project_dir = Path(__file__).resolve().parent.parent


def treeIso(
    treeName,
    depth_threshold=120,
    colour_threshold=18,
    dilation_iterations=5,
    ground_threshold=0.7
):
    """
    Isolate an apple tree from winter RGB and depth observations.

    The algorithm combines depth-based segmentation with colour
    information to identify tree pixels. Morphological dilation
    allows nearby colour-detected pixels to be included, while
    row-based ground removal and connected-component analysis
    attempt to separate the tree from surrounding objects.

    Args:
        treeName (str): Unique tree identifier in the dataset,
            such as "R01N01".
        depth_threshold (int, optional): Minimum encoded depth
            value required for a pixel to be selected.
            Defaults to 120.
        colour_threshold (int, optional): Minimum difference
            between the blue and green channels (B - G)
            required for colour segmentation. Defaults to 18.
        dilation_iterations (int, optional): Number of binary
            dilation iterations applied to the depth mask.
            Defaults to 5.
        ground_threshold (float, optional): Fraction of foreground
            pixels in an image row above which the entire row
            is classified as ground. Defaults to 0.7.

    Returns:
        np.ndarray: Two-dimensional Boolean array with the same
            height and width as the winter depth image. True indicates
            a pixel classified as part of the isolated tree; False
            indicates background.

    Raises:
        ValueError: If the requested tree identifier is not present
            in the dataset, or if required winter images are missing.
        FileNotFoundError: If the dataset directory does not exist,
            when directory validation is enabled in TreeDataset.
        OSError: If an image file cannot be opened or decoded.

    Example:
        >>> mask = treeIso("R01N01")
        >>> mask = treeIso(
        ...     "R01N01",
        ...     depth_threshold=130,
        ...     colour_threshold=20
        ... )

    Notes:
        The algorithm uses 8-neighbour connectivity and retains
        the largest connected region after ground removal.

        The selected thresholds are empirical and may not be
        optimal for all trees. The returned mask is a segmentation
        result, not a calibrated 3D reconstruction.

        If no foreground component remains after ground removal,
        an all-False mask is returned.
    """

    # Locate the raw image dataset within the project directory.
    data_dir = project_dir / "data/raw/Dataset (With Summer GT)"

    # Access the selected tree through the dataset management class.
    dataset = TreeDataset(data_dir)
    tree = dataset.get_tree(treeName)

    # Check that both winter observations are available before processing.
    # Missing images would otherwise cause an unclear indexing error.
    if tree.winter_depth is None or tree.winter_rgb is None:
        raise ValueError(
            f"Missing winter RGB or depth image for tree '{treeName}'."
        )

    # Extract one depth channel and the corresponding winter RGB image.
    # The depth channels are assumed to contain identical information.
    depth = tree.winter_depth[:, :, 0]
    rgb = tree.winter_rgb

    # --------------------------------------------------
    # DEPTH THRESHOLD
    # --------------------------------------------------

    # Identify candidate tree pixels using encoded depth values.
    # The threshold is empirical and does not represent a physical distance.
    tree_mask = depth > depth_threshold

    # --------------------------------------------------
    # COLOUR THRESHOLD
    # --------------------------------------------------

    # Extract the green and blue channels for colour-based segmentation.
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # Identify pixels where blue intensity exceeds green intensity.
    # Convert to signed integers to avoid uint8 subtraction underflow.
    colour_mask = (
        B.astype(int) - G.astype(int)
    ) > colour_threshold

    # --------------------------------------------------
    # DILATION
    # --------------------------------------------------

    # Expand the depth mask to include nearby pixels that may
    # represent thin branches missed by depth thresholding.
    expanded_depth = binary_dilation(
        tree_mask,
        iterations=dilation_iterations
    )

    # Restrict colour-detected pixels to the expanded depth region.
    # This reduces the inclusion of distant background objects.
    connected_colour = colour_mask & expanded_depth

    # Combine the original depth candidates with nearby colour candidates.
    combined_mask = tree_mask | connected_colour

    # --------------------------------------------------
    # REMOVE GROUND
    # --------------------------------------------------

    # Estimate the proportion of foreground pixels in each image row.
    row_counts = np.sum(combined_mask, axis=1)
    row_fraction = row_counts / combined_mask.shape[1]

    # Classify rows with high foreground coverage as potential ground.
    # This assumes ground occupies a broad horizontal region.
    ground_rows = row_fraction > ground_threshold

    # Remove classified ground rows before selecting the final component.
    # Dense horizontal branches may also be removed by this operation.
    ground_removed = combined_mask.copy()
    ground_removed[ground_rows, :] = False

    # --------------------------------------------------
    # FIND CONNECTED GROUPS
    # --------------------------------------------------

    binary_tree = ground_removed

    # Identify connected foreground regions after ground removal.
    # This allows regions previously connected through the ground
    # to be treated as separate components.
    visited = np.zeros(binary_tree.shape, dtype=bool)
    groups = []

    rows, cols = binary_tree.shape

    # Search each pixel for the beginning of an unvisited component.
    for row in range(rows):
        for col in range(cols):

            if binary_tree[row, col] and not visited[row, col]:

                group = []
                stack = [(row, col)]

                # Use an iterative depth-first search to identify
                # all connected foreground pixels in this component.
                while stack:
                    r, c = stack.pop()

                    if visited[r, c]:
                        continue

                    visited[r, c] = True

                    if not binary_tree[r, c]:
                        continue

                    group.append((r, c))

                    # Examine all eight neighbouring pixel positions.
                    # Image boundaries are checked to prevent invalid indices.
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:

                            nr = r + dr
                            nc = c + dc

                            if (0 <= nr < rows and
                                0 <= nc < cols and
                                not visited[nr, nc]):

                                stack.append((nr, nc))

                groups.append(group)

    # --------------------------------------------------
    # KEEP LARGEST CONNECTED GROUP
    # --------------------------------------------------

    # Initialise an empty output mask in case no components remain.
    main_tree_mask = np.zeros(binary_tree.shape, dtype=bool)

    if groups:
        # Retain the largest remaining connected component,
        # assuming it represents the main tree structure.
        largest_group = max(groups, key=len)

        # Mark the pixels belonging to the selected component.
        for row, col in largest_group:
            main_tree_mask[row, col] = True

    # Return the final Boolean tree segmentation mask.
    return main_tree_mask
