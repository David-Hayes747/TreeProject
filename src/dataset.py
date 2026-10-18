
import numpy as np
from pathlib import Path
from PIL import Image


class Tree:
    """
    Stores and provides access to the images associated with an individual tree.

    Each tree can have five observations: summer RGB, summer depth,
    summer ground truth, winter RGB, and winter depth. The corresponding
    image paths and NumPy arrays are stored as attributes.

    Attributes:
        tree_id (str): Unique identifier for the tree.
        data_dir (Path): Directory containing the dataset images.
        files (list[Path]): PNG files associated with the tree.
        summer_rgb (np.ndarray or None): Summer RGB image.
        summer_depth (np.ndarray or None): Summer depth image.
        summer_gt (np.ndarray or None): Summer ground-truth image.
        winter_rgb (np.ndarray or None): Winter RGB image.
        winter_depth (np.ndarray or None): Winter depth image.

    Note:
        Missing images are stored as None. Image data is loaded
        without modification or conversion of pixel values.
    """

    def __init__(self, tree_id, data_dir):
        """
        Initialise a tree and load its available observations.

        Args:
            tree_id (str): Unique tree identifier used in image filenames.
            data_dir (str or Path): Directory containing the dataset images.

        Raises:
            OSError: If an existing image file cannot be opened or read.

        Example:
            >>> tree = Tree("R01N01", "data/raw/Dataset (With Summer GT)")
            >>> rgb = tree.winter_rgb
        """
        self.tree_id = tree_id
        self.data_dir = Path(data_dir)

        # Identify the image files associated with the selected tree ID.
        self.files = list(self.data_dir.glob(f"{tree_id}_*.png"))

        # Locate the five expected observation types using filename keywords.
        # A missing observation is represented by a None path.
        self.summer_rgb_path = self.find_file("Summer_RGB")
        self.summer_depth_path = self.find_file("Summer_D")
        self.summer_gt_path = self.find_file("Summer_GT")

        self.winter_rgb_path = self.find_file("Winter_RGB")
        self.winter_depth_path = self.find_file("Winter_D")

        # Load observations into NumPy arrays for subsequent image processing.
        # Missing observations remain None.
        self.summer_rgb = self.load_image(self.summer_rgb_path)
        self.summer_depth = self.load_image(self.summer_depth_path)
        self.summer_gt = self.load_image(self.summer_gt_path)

        self.winter_rgb = self.load_image(self.winter_rgb_path)
        self.winter_depth = self.load_image(self.winter_depth_path)


    def find_file(self, keyword):
        """
        Find an image file associated with the tree using a filename keyword.

        Args:
            keyword (str): Text identifying the required observation type,
                such as "Winter_RGB" or "Summer_GT".

        Returns:
            Path or None: Path to the first matching image file,
            or None if no matching file exists.

        Note:
            Matching is based on whether the keyword occurs in the
            filename. If multiple files match, the first is returned.
        """
        for file in self.files:
            if keyword in file.name:
                return file

        return None


    def load_image(self, path):
        """
        Load an image from disk as a NumPy array.

        Args:
            path (str or Path or None): Path to the image file.
                None indicates that the observation is unavailable.

        Returns:
            np.ndarray or None: Array containing the original image
            pixel values, or None if no path was provided.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            OSError: If the image cannot be opened or decoded.

        Note:
            No resizing, normalisation, or depth conversion is performed.
            Image dimensions and data types depend on the original file.
        """
        if path is None:
            return None

        return np.array(Image.open(path))


class TreeDataset:
    """
    Manages the dataset and provides access to individual tree observations.

    The dataset is organised using unique tree identifiers extracted
    from the filenames. This class allows trees to be selected without
    manually specifying the paths to their individual images.

    Attributes:
        data_dir (Path): Directory containing the dataset images.
        file_list (list[Path]): All PNG files in the dataset directory.
        tree_names (list[str]): Sorted list of unique tree identifiers.
        number_of_images (int): Total number of PNG files.
        number_of_trees (int): Total number of unique tree identifiers.

    Note:
        Tree identifiers are assumed to occupy the first six characters
        of each filename. Only PNG files directly inside the specified
        directory are included.
    """

    def __init__(self, data_dir):
        """
        Initialise the dataset and identify all available trees.

        Args:
            data_dir (str or Path): Directory containing the dataset
                PNG images.

        Note:
            This method indexes filenames but does not load image data.
            Individual images are loaded when get_tree() is called.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                "Dataset directory not found: {self.data_dir}"
            )

        # Index PNG files without loading their image data into memory.
        self.file_list = list(self.data_dir.glob("*.png"))

        # Extract unique six-character tree IDs from the dataset filenames.
        self.tree_names = sorted(
            set(file.name[:6] for file in self.file_list)
        )

        # Store dataset statistics for exploration and completeness checks.
        self.number_of_images = len(self.file_list)
        self.number_of_trees = len(self.tree_names)


    def get_tree(self, tree_id):
        """
        Retrieve a tree and load its available image observations.

        Args:
            tree_id (str): Unique identifier of the requested tree,
                such as "R01N01".

        Returns:
            Tree: Object containing the image paths and NumPy arrays
            associated with the requested tree.

        Raises:
            ValueError: If the requested tree identifier is not present
                in the dataset.
            OSError: If an associated image cannot be opened or read.

        Example:
            >>> dataset = TreeDataset("data/raw/Dataset (With Summer GT)")
            >>> tree = dataset.get_tree("R01N01")
            >>> winter_image = tree.winter_rgb
        """
        if tree_id not in self.tree_names:
            raise ValueError(f"Tree '{tree_id}' not found in dataset.")

        return Tree(tree_id, self.data_dir)
