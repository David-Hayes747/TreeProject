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

The optimised parameters increased the mean IoU by 6.84