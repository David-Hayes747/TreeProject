# MMA3001 – Apple Tree Isolation Project

## Overview

This repository contains my individual project for MMA3001 – Numerical Methods and Machine Learning at Monash University.

The project investigates the use of image processing techniques to isolate the main structure of apple trees from RGB and depth images.

The aim was to develop a Python-based algorithm capable of identifying the trunk and branches of a tree from winter observations while removing surrounding objects such as neighbouring trees, vegetation and the ground.

The final algorithm combines depth thresholding, colour segmentation, morphological dilation, ground removal and connected component analysis to produce a two-dimensional binary mask of the tree.

The algorithm was evaluated against the provided summer ground-truth images, and its parameters were optimised to improve the measured segmentation accuracy.

## Engineering Problem

Identifying specific objects and structures within images is an important problem in computer vision, with applications in autonomous vehicles, robotics and environmental monitoring.

In this project, the objective was to isolate the main structure of an apple tree from winter RGB-D observations.

This was challenging because the tree could have similar depth and colour values to surrounding objects. Neighbouring trees, branches and the ground could also be present in the images, making it difficult to identify only the tree being investigated.

The project focused on developing and evaluating a two-dimensional tree isolation algorithm rather than reconstructing a complete three-dimensional tree.

## Dataset

The dataset used in this project is the **apple tree dataset provided through the MMA3001 Google Drive** at Monash University.

The dataset contains observations of 562 apple trees, with images captured during summer and winter.

Each tree has the following data:

- Summer RGB image
- Summer depth image
- Summer ground-truth (GT) mask
- Winter RGB image
- Winter depth image

The RGB and depth images have dimensions of 480 × 640 pixels, while the ground-truth masks have dimensions of 256 × 256 pixels.

The winter RGB and depth images were used as the primary inputs for the tree isolation algorithm.

The summer ground-truth masks were used to evaluate the results. However, the summer and winter observations were captured from different viewpoints, meaning that alignment was required before comparing them.

### Dataset Location

The dataset can be obtained from the MMA3001 Google Drive provided to students enrolled in the unit.

Due to its large size, the dataset is not included in this repository.

After obtaining the dataset, place it in the following directory:

```text
data/
└── raw/
    └── Dataset (With Summer GT)/
        ├── R01N01_Summer_RGB.png
        ├── R01N01_Summer_D.png
        ├── R01N01_Summer_GT.png
        ├── R01N01_Winter_RGB.png
        ├── R01N01_Winter_D.png
        └── ...
```

The folder name and file naming convention should be maintained so that the Python functions can correctly locate the images.

## Dependencies and Installation

The project was developed using Python and Jupyter notebooks.

The main Python libraries used are:

| Library | Purpose |
|---|---|
| NumPy | Numerical operations and image arrays |
| SciPy | Morphological image processing |
| Pillow (PIL) | Image loading and manipulation |
| Matplotlib | Visualisation of images, masks and results |
| Jupyter | Development, experimentation and evaluation notebooks |
| pdoc | Generation of HTML documentation |

### Installation

1. Download or clone the GitHub repository.
2. Open the project folder in VS Code or another Python development environment.
3. Ensure Python is installed.
4. Install the required libraries using:

```bash
python -m pip install numpy scipy pillow matplotlib jupyter pdoc
```

5. Download the tree dataset from the MMA3001 Google Drive and place it in the directory described above.

The project can then be run using the Python main file or explored through the Jupyter notebooks.

## Repository Structure

```text
TreeProject/
│
├── data/
│   └── raw/
│       └── Dataset (With Summer GT)/
│
├── docs/
│   ├── index.html
│   ├── src.html
│   └── src/
│       ├── dataset.html
│       ├── tree_isolation.html
│       ├── tree_alignment.html
│       └── evaluation.html
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tree_reconstruction.ipynb
│   ├── 03_result_comparison.ipynb
│   └── 04_parameter_optimisation.ipynb
│
├── results/
│   └── parameter_results.csv
│
├── src/
│   ├── dataset.py
│   ├── tree_isolation.py
│   ├── tree_alignment.py
│   └── evaluation.py
│
├── MainFile.py
├── README.md
└── LICENSE
```

### Source Files

**dataset.py**

Contains the `Tree` and `TreeDataset` classes, which manage the dataset and provide access to the RGB, depth and ground-truth images associated with each tree.

**tree_isolation.py**

Contains the main tree isolation function, which combines depth thresholding, colour segmentation, morphological dilation, ground removal and connected component analysis.

**tree_alignment.py**

Contains functions used to align the summer ground-truth masks with the predicted winter tree masks. This includes resizing, V-junction identification and image translation.

**evaluation.py**

Contains the functions used to calculate the evaluation metrics, including Intersection over Union, Dice coefficient, precision and recall.

## Running the Project

The main algorithm can be run using `MainFile.py`.

From the main project directory, run:

```bash
python MainFile.py
```

The program will request a tree ID in the format:

```text
R01N03
```

The corresponding winter RGB and depth images will be loaded from the dataset.

The tree isolation algorithm will then process the images and produce a binary mask showing the isolated tree.

The output is displayed using Matplotlib.

### Algorithm Parameters

The main algorithm allows four parameters to be adjusted:

| Parameter | Original Value | Description |
|---|---|---|
| depth_threshold | 120 | Minimum depth value used to identify potential tree pixels |
| colour_threshold | 18 | Minimum blue-green channel difference |
| dilation_iterations | 5 | Number of morphological dilation iterations |
| ground_threshold | 0.7 | Proportion of foreground pixels required to remove a horizontal row |

These parameters can be modified when calling the `treeIso()` function.

## Computational Approach

The tree isolation algorithm was developed using several image processing techniques.

### 1. Depth Thresholding

The first stage uses the winter depth image to identify pixels that potentially belong to the tree.

A threshold is applied to the depth values, producing a binary mask containing pixels above the selected threshold.

The original depth threshold was 120.

Although this successfully identified parts of the tree, it also included other objects at similar depths.

### 2. Colour Segmentation

The winter RGB image is used to further distinguish the tree from its surroundings.

The algorithm compares the blue and green colour channels using the difference:

`Blue - Green`

Pixels with a difference greater than the selected colour threshold are identified as potential tree pixels.

The original colour threshold was 18.

The colour mask is combined with the depth mask to improve the identification of the tree.

### 3. Morphological Dilation

Morphological dilation is applied to expand the detected regions and reconnect sections of the tree that may have become separated during thresholding.

This helps retain the main branch structure before connected component analysis.

### 4. Ground Removal

The ground often has similar depth and colour values to the tree, causing it to be included in the initial segmentation.

To address this, the algorithm examines the proportion of foreground pixels in each horizontal row.

Rows exceeding a specified foreground proportion are removed.

The original ground threshold was 0.7, which was later reduced to 0.5 through optimisation.

### 5. Connected Component Analysis

Connected component analysis is used to identify the main tree from the remaining foreground pixels.

The algorithm uses 8-neighbour connectivity, where pixels are considered connected horizontally, vertically or diagonally.

The largest connected region is selected as the main tree, while smaller disconnected regions are removed.

This approach assumes that the main tree forms the largest connected foreground region.

## Outputs

The main output is a two-dimensional binary mask with dimensions of 480 × 640 pixels.

Each pixel contains a Boolean value indicating whether it belongs to the isolated tree.

- `True` – Pixel identified as part of the tree.
- `False` – Pixel identified as background.

The output can be visualised as a black-and-white image showing the isolated trunk and branches.

## Validation

### Ground-Truth Alignment

The provided ground-truth masks were captured during summer and differed in size, orientation and viewpoint from the winter observations.

To allow quantitative comparison, an automatic alignment method was developed.

The method:

1. Resizes the ground-truth image using a scaling factor of 0.75.
2. Identifies the approximate V-shaped junction where the main branches separate.
3. Locates the corresponding junction in the isolated tree mask.
4. Translates the ground-truth image to align the two junctions.

This provided an approximate alignment between the predicted tree mask and the ground truth.

However, the method could not fully account for differences in rotation, camera perspective or visible branch structure.

### Evaluation Metrics

Four evaluation metrics were used:

- **Intersection over Union (IoU):** Measures the overlap between the predicted tree mask and ground truth relative to their combined area.
- **Dice Coefficient:** Measures similarity between the predicted and ground-truth masks.
- **Precision:** Measures the proportion of predicted tree pixels that are correctly identified.
- **Recall:** Measures the proportion of ground-truth tree pixels successfully identified.

### Initial Validation Results

The algorithm was evaluated across the dataset of 562 trees using the original parameter values.

Of these, 422 trees were successfully evaluated, while 140 did not produce valid evaluation results.

| Metric | Mean | Standard Deviation |
|---|---|---|
| IoU | 0.2019 | 0.0957 |
| Dice Coefficient | 0.3250 | 0.1382 |
| Precision | 0.2695 | 0.1355 |
| Recall | 0.4495 | 0.1608 |

The results indicate that the algorithm was capable of identifying tree structures, although the measured agreement with the aligned ground truth was relatively low.

The evaluation results were also affected by inaccuracies in the ground-truth alignment.

## Parameter Optimisation

Four parameters were investigated to improve the segmentation accuracy:

1. Depth threshold
2. Colour threshold
3. Dilation iterations
4. Ground removal threshold

Each parameter was varied independently while the remaining parameters were held constant.

The mean IoU was used to identify the best-performing parameter values.

### Final Parameter Selection

| Parameter | Original | Optimised |
|---|---|---|
| Depth threshold | 120 | 125 |
| Colour threshold | 18 | 18 |
| Dilation iterations | 5 | 4 |
| Ground threshold | 0.7 | 0.5 |

### Optimisation Results

The original and optimised configurations were compared using the first 40 trees in the dataset.

Of these, 29 trees were successfully evaluated under both configurations.

| Evaluation | Mean IoU |
|---|---|
| Original algorithm | 0.1825 |
| Optimised algorithm | 0.1950 |
| Relative improvement | +6.84% |

The optimised parameters increased the mean IoU by 6.84%, from 0.1825 to 0.1950.

Although the improvement was relatively small, it demonstrated that quantitative parameter selection could improve the measured performance of the algorithm.

The overall IoU remained low, partly due to the limitations of the ground-truth alignment method.

The optimisation also investigated each parameter independently rather than testing all possible combinations. Therefore, the final configuration cannot be guaranteed to be globally optimal.

## Development, Testing and Evaluation Notebooks

The Jupyter notebooks document the development of the project in chronological order.

They contain the initial exploration of the dataset, development and experimentation with computational methods, comparison of results and parameter optimisation.

### 01_data_exploration.ipynb

Contains the initial exploration of the provided dataset, including examination of the available images and their characteristics.

### 02_tree_reconstruction.ipynb

Documents the development of the tree isolation approach and experimentation with image-processing methods.

### 03_result_comparison.ipynb

Contains the comparison of predicted tree masks with the available ground-truth data, including alignment and quantitative evaluation.

### 04_parameter_optimisation.ipynb

Investigates the effects of different algorithm parameters and compares the original and optimised configurations.

### Testing and Verification

The notebooks contain the development experiments and evaluation work used to assess the algorithm.

Testing and verification activities include:

- Visual inspection of RGB, depth and segmented tree images.
- Examination of intermediate masks produced by the image-processing methods.
- Comparison of predicted masks against aligned ground-truth masks.
- Calculation of IoU, Dice coefficient, precision and recall.
- Investigation of parameter sensitivity.
- Comparison of original and optimised algorithm configurations.
- Identification and discussion of unsuccessful evaluations.

These investigations were used to identify problems, evaluate computational decisions and improve the algorithm during development.

The notebooks retain the progression of the project, allowing the development and evaluation process to be reviewed.

The quantitative validation results are discussed in the project report.

## HTML Documentation

HTML documentation has been generated using `pdoc` from the Python docstrings.

The documentation is located in the `docs/` directory and includes descriptions of the source modules, classes and functions.

To view the documentation, open:

```text
docs/index.html
```

in a web browser.

To regenerate the documentation after modifying the source code, run:

```bash
python -m pdoc -o docs src
```

The generated documentation provides additional information about the implementation, including function parameters and return values.

## Limitations

Several limitations were identified during development and evaluation.

### Ground-Truth Alignment

The summer ground-truth images and winter observations were captured from different viewpoints.

The single-point alignment method could not fully correct these differences, affecting the reliability of the calculated evaluation metrics.

### Fixed Thresholds

The depth and colour thresholds were constant across the dataset.

Different lighting conditions, backgrounds and tree characteristics could reduce the effectiveness of these fixed values.

### Connected Component Assumptions

The algorithm assumes that the main tree is the largest connected foreground region.

If another object or neighbouring tree forms a larger region, the algorithm may select the wrong object.

### Ground Removal

The ground removal method could incorrectly remove parts of the tree if large horizontal branches were present.

### Evaluation Limitations

Not all trees could be successfully evaluated.

The optimisation was also performed using observations from the same dataset used during development, so its performance on unseen observations has not been independently established.

## Lessons Learned

The project demonstrated the importance of combining multiple computational methods to solve a problem that could not be addressed reliably using a single technique.

It also highlighted the importance of validation and the limitations of using ground-truth data that does not directly correspond to the observations being evaluated.

Quantitative parameter optimisation allowed computational decisions to be supported by measured results rather than relying entirely on visual inspection.

The project also provided experience with Python programming, image processing, dataset management, computational experimentation and numerical evaluation.

## AI Usage

OpenAI's ChatGPT was used during the project to assist with Python programming, troubleshooting errors, exploring computational methods, improving documentation and editing the written report.

AI-generated suggestions were tested and evaluated before being incorporated into the project.

The final computational approach, parameter selection, validation and interpretation of results remained my responsibility.

A more detailed AI-use reflection is included in the project report.

## Licence

This project is distributed under the MIT License.

See the `LICENSE` file for details.

The apple tree dataset was provided separately through MMA3001 and is not covered by the repository's software licence.

## Conclusion

This project developed a Python-based algorithm capable of isolating the main structure of apple trees from winter RGB and depth images.

By combining depth thresholding, colour segmentation, morphological dilation, ground removal and connected component analysis, the algorithm produced two-dimensional binary masks identifying the tree structure.

Quantitative evaluation and parameter optimisation demonstrated measurable improvements in the results, while also identifying limitations associated with ground-truth alignment and fixed segmentation parameters.

The project demonstrates the application of numerical methods, image processing and computational evaluation to an engineering problem.