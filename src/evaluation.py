import numpy as np

def calculate_metrics(prediction, gt):

    prediction = np.asarray(prediction).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if prediction.shape != gt.shape:
        raise ValueError("Prediction and GT must have the same shape")

    tp = np.sum(prediction & gt)
    fp = np.sum(prediction & ~gt)
    fn = np.sum(~prediction & gt)

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