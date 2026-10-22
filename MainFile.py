# Import libraries
import matplotlib.pyplot as plt

import sys
from pathlib import Path

project_dir = Path("..").resolve()

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from src.tree_isolation import treeIso

tree_id = "R01N03"
tree_id = input("Name of the tree in the format R##N##")

isolatedTree = treeIso(
    tree_id,
    depth_threshold=120,
    colour_threshold=18,
    dilation_iterations=5,
    ground_threshold=0.7
)
print (isolatedTree)

plt.figure(figsize=(8, 8))
plt.imshow(isolatedTree, cmap="gray")
plt.title("Isolated Tree Mask")
plt.axis("off")
plt.show()