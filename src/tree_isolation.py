# Import libraries

import numpy as np
from pathlib import Path
from .dataset import TreeDataset

# Add the main project folder to Python's search path
project_dir = Path("..").resolve()



def treeIso(treeName):
    # Dataset location
    data_dir = project_dir / "data/raw/Dataset (With Summer GT)"

    # Create dataset
    dataset = TreeDataset(data_dir)

    # Select one tree for initial reconstruction
    tree = dataset.get_tree(treeName)
    depth = tree.winter_depth[:, :, 0]
    rgb = tree.winter_rgb

    depth_threshold = 120
    tree_mask = depth > depth_threshold

    tree_depth_values = depth[tree_mask]

    tree_depth = np.zeros_like(depth)
    tree_depth[tree_mask] = depth[tree_mask]
    # Find all non-zero pixels
    binary_tree = tree_depth > 0

    # Keep track of pixels already checked
    visited = np.zeros(binary_tree.shape, dtype=bool)

    groups = []

    rows, cols = binary_tree.shape

    # Check every pixel
    for row in range(rows):
        for col in range(cols):

            # Start a new group if this pixel is non-zero and hasn't been checked
            if binary_tree[row, col] and not visited[row, col]:

                group = []
                stack = [(row, col)]

                while stack:
                    r, c = stack.pop()

                    if visited[r, c]:
                        continue

                    visited[r, c] = True

                    if not binary_tree[r, c]:
                        continue

                    group.append((r, c))

                    # Check all 8 neighbouring pixels
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:

                            nr = r + dr
                            nc = c + dc

                            if (0 <= nr < rows and
                                0 <= nc < cols and
                                not visited[nr, nc]):

                                stack.append((nr, nc))

                groups.append(group)

    # Find largest connected group
    largest_group = max(groups, key=len)

    print("Number of groups:", len(groups))
    print("Size of largest group:", len(largest_group))

    # Create mask for largest group
    main_tree_mask = np.zeros(binary_tree.shape, dtype=bool)
    print("main_tree_mask")

    for row, col in largest_group:
        main_tree_mask[row, col] = True

    # Keep depth values only for main tree
    main_tree_depth = np.where(main_tree_mask, depth, 0)

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    colour_mask = (B.astype(int) - G.astype(int)) > 18
    from scipy.ndimage import binary_dilation
    expanded_depth = binary_dilation(tree_mask, iterations=5)
    connected_colour = colour_mask & expanded_depth
    combined_mask = tree_mask | connected_colour


    # --------------------------------------------------
    # REMOVE GROUND
    # --------------------------------------------------

    # Count how many detected pixels are in each row
    row_counts = np.sum(combined_mask, axis=1)

    # Find the fraction of each row that is detected
    row_fraction = row_counts / combined_mask.shape[1]

    # Rows where more than 70% of pixels are detected are assumed to be ground
    ground_rows = row_fraction > 0.70

    # Copy the combined mask and remove the ground rows
    ground_removed = combined_mask.copy()
    ground_removed[ground_rows, :] = False


    # --------------------------------------------------
    # FIND CONNECTED GROUPS
    # --------------------------------------------------

    binary_tree = ground_removed

    # Keep track of pixels already checked
    visited = np.zeros(binary_tree.shape, dtype=bool)

    groups = []

    rows, cols = binary_tree.shape

    # Check every pixel
    for row in range(rows):
        for col in range(cols):

            # Start a new group if this pixel is white and hasn't been checked
            if binary_tree[row, col] and not visited[row, col]:

                group = []
                stack = [(row, col)]

                while stack:
                    r, c = stack.pop()

                    if visited[r, c]:
                        continue

                    visited[r, c] = True

                    if not binary_tree[r, c]:
                        continue

                    group.append((r, c))

                    # Check all 8 neighbouring pixels
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

    largest_group = max(groups, key=len)

    # Create an empty mask
    main_tree_mask = np.zeros(binary_tree.shape, dtype=bool)

    # Add the largest group to the mask
    for row, col in largest_group:
        main_tree_mask[row, col] = True
    print(main_tree_mask[row,col])
    return main_tree_mask




