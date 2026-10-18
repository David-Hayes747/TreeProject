
"""
Evaluation metrics for comparing predicted tree masks with ground truth.

Provides pixel-level segmentation metrics including Intersection over
Union (IoU), Dice coefficient, precision and recall.
"""

import numpy as np


def calculate_metrics(prediction, gt):
    """
    Calculate segmentation performance against a ground-truth mask.

    Both masks are converted to Boolean arrays before comparing
    corresponding pixels.

    Args:
        prediction (np.ndarray): Predicted binary segmentation mask.
        gt (np.ndarray): Ground-truth binary segmentation mask.

    Returns:
        dict: Dictionary containing:
            - IoU (float): Intersection over Union.
            - Dice (float): Dice similarity coefficient.
            - Precision (float): Proportion of predicted foreground
              pixels that are correctly classified.
            - Recall (float): Proportion of ground-truth foreground
              pixels correctly identified.

        Undefined metrics are returned as np.nan when their
        denominators are zero.

    Raises:
        ValueError: If the prediction and ground-truth masks
            have different dimensions.
    """

    # Convert both inputs to Boolean masks.
    prediction = np.asarray(prediction).astype(bool)
    gt = np.asarray(gt).astype(bool)

    # Pixel-wise comparison requires matching image dimensions.
    if prediction.shape != gt.shape:
        raise ValueError("Prediction and GT must have the same shape")

    # Count true positives, false positives and false negatives.
    tp = np.sum(prediction & gt)
    fp = np.sum(prediction & ~gt)
    fn = np.sum(~prediction & gt)

    # Calculate segmentation metrics.
    # Return NaN where the metric is mathematically undefined.
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall
    }
