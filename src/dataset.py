import numpy as np
from pathlib import Path
from PIL import Image


class Tree:
    """
    Stores the RGB, depth and ground-truth data associated with one tree.
    """

    def __init__(self, tree_id, data_dir):
        self.tree_id = tree_id
        self.data_dir = Path(data_dir)

        # Find all files belonging to this tree
        self.files = list(self.data_dir.glob(f"{tree_id}_*.png"))

        # Find paths
        self.summer_rgb_path = self.find_file("Summer_RGB")
        self.summer_depth_path = self.find_file("Summer_D")
        self.summer_gt_path = self.find_file("Summer_GT")

        self.winter_rgb_path = self.find_file("Winter_RGB")
        self.winter_depth_path = self.find_file("Winter_D")

        # Load images
        self.summer_rgb = self.load_image(self.summer_rgb_path)
        self.summer_depth = self.load_image(self.summer_depth_path)
        self.summer_gt = self.load_image(self.summer_gt_path)

        self.winter_rgb = self.load_image(self.winter_rgb_path)
        self.winter_depth = self.load_image(self.winter_depth_path)


    def find_file(self, keyword):
        """Find a file belonging to the tree containing a keyword."""

        for file in self.files:
            if keyword in file.name:
                return file

        return None


    def load_image(self, path):
        """Load an image and return it as a NumPy array."""

        if path is None:
            return None

        return np.array(Image.open(path))


class TreeDataset:
    """
    Provides access to all trees contained within the dataset.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # Find all PNG files
        self.file_list = list(self.data_dir.glob("*.png"))

        # Determine unique tree IDs
        self.tree_names = sorted(
            set(file.name[:6] for file in self.file_list)
        )

        self.number_of_images = len(self.file_list)
        self.number_of_trees = len(self.tree_names)


    def get_tree(self, tree_id):
        """Return the requested Tree object."""

        if tree_id not in self.tree_names:
            raise ValueError(f"Tree '{tree_id}' not found in dataset.")

        return Tree(tree_id, self.data_dir)