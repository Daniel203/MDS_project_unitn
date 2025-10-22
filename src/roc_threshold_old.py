"""
Threshold estimation using ROC curves for spread‑spectrum watermark detection.

This script provides a helper function to compute an appropriate
similarity threshold τ for the non‑blind detector implemented in
``detection_groupname.py``.  The procedure follows the guidelines from
the laboratory: embed your watermark into a set of representative
images, attack them using one or more processing operations, extract
the embedded information from the attacked images and compare it with
the known watermark bits.  Similarity scores for both true positives
and true negatives are collected and a Receiver Operating
Characteristic (ROC) curve is calculated.  A threshold is then chosen
such that the false positive rate (FPR) does not exceed the specified
limit (e.g. 0.1).  The returned τ should be substituted into the
``TAU`` constant in the detection module.

The example below demonstrates how to use this module:

.. code-block:: python

   images = ['lena_grey.bmp', 'some_other.bmp', ...]
   tau = compute_threshold('mark.npy', images, attack_name='awgn',
                           attack_params=[20], fpr_limit=0.1)
   print(f"Estimated threshold: {tau:.4f}")

This code uses scikit‑learn's :func:`roc_curve` to compute ROC
characteristics.  Ensure that scikit‑learn is available in your
environment.  The selected threshold maximises the detection margin
while respecting the FPR constraint.
"""

from __future__ import annotations
from typing import List, Sequence
import os
import tempfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from embedding import embedding
from attacks import attacks as apply_attacks
import matplotlib.pyplot as plt




def compute_threshold(watermark_path: str,
                      image_paths: Sequence[str],
                      attack_name: str = 'awgn',
                      attack_params: Sequence = (10,),
                      fpr_limit: float = 0.1) -> float:
    """Compute a similarity threshold via ROC analysis.

    Args:
        watermark_path: Path to the NumPy file containing the original
            watermark bits (length 1024).  Values must be 0 or 1.
        image_paths: A sequence of paths to images used to estimate the
            threshold.  These images should be representative of the
            content you expect to watermark and attack.
        attack_name: Name of the attack to use when generating examples.
            See :func:`attacks` for valid names.  If multiple attacks
            should be applied, you can combine them externally.
        attack_params: Parameters for the chosen attack.  For multiple
            attacks supply a sequence of sequences matching the names.
        fpr_limit: Maximum acceptable false positive rate when selecting
            the threshold.

    Returns:
        The estimated threshold τ on the similarity measure.
    """
    # Load the watermark bits once and map them to 0/1.
    watermark = np.load(watermark_path).astype(np.uint8).flatten()
    if watermark.size != 1024:
        raise ValueError("Watermark must have length 1024")

    scores: List[float] = []
    labels: List[int] = []

    for img_path in image_paths:
        # Embed the watermark into the host image
        wm_img = embedding(img_path, watermark_path)
        # Read the original image
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        # Create a temporary file for the watermarked image so that
        # attacks() can operate on a filename.  Note: NamedTemporaryFile
        # with delete=False is used to persist the file for reading.
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, wm_img)
        try:
            attacked = apply_attacks(temp_path, attack_name, attack_params)
        finally:
            # Always remove the temporary file
            os.remove(temp_path)

        # Convert images to float32 and compute DCT of original and attacked
        dct_orig = cv2.dct(orig_img.astype(np.float32))
        dct_att = cv2.dct(attacked.astype(np.float32))
        # Determine the embedding positions by sorting the original DCT
        flat = dct_orig.ravel()
        sorted_idx = np.argsort(np.abs(flat))[::-1]
        embed_idx = sorted_idx[1:1025]
        # Extract bits from attacked image relative to original
        att_bits = np.zeros(1024, dtype=np.uint8)
        for i, idx in enumerate(embed_idx):
            r, c = divmod(int(idx), dct_orig.shape[1])
            att_bits[i] = 1 if (dct_att[r, c] - dct_orig[r, c]) >= 0 else 0
        # True positive: compare extracted bits with the actual watermark
        tp_score = float(np.sum(att_bits == watermark)) / len(watermark)
        scores.append(tp_score)
        labels.append(1)
        # True negative: compare extracted bits with a random watermark
        rand_bits = np.random.randint(0, 2, 1024, dtype=np.uint8)
        tn_score = float(np.sum(att_bits == rand_bits)) / len(rand_bits)
        scores.append(tn_score)
        labels.append(0)

    # Compute ROC curve using the collected scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Identify thresholds with FPR below the limit
    valid = thresholds[fpr <= fpr_limit]
    if valid.size == 0:
        # No threshold satisfies the FPR constraint; choose the most
        # conservative threshold (minimise false positives)
        tau = float(thresholds[np.argmin(fpr)])
    else:
        tau = float(np.max(valid))

    plot_roc_curve(labels, scores)
    
    return tau


def compute_roc_curve(watermark_path: str,
                      image_paths: Sequence[str],
                      attack_name: str = 'awgn',
                      attack_params: Sequence = (10,),
                      fpr_limit: float = 0.1,
                      n_samples: int = 100,
                      plot_file: str = 'roc_curve.png') -> float:
    """Compute and plot a ROC curve for watermark detection.

    This function extends :func:`compute_threshold` by repeatedly attacking
    each watermarked image and generating multiple similarity scores.
    It uses the spread‑spectrum detection logic implemented in
    ``detection_groupname.py``.  A ROC curve is computed via
    scikit‑learn and a figure is saved to ``plot_file``.

    Args:
        watermark_path: Path to the NumPy file containing the original
            watermark bits (length 1024).  Values must be 0 or 1.
        image_paths: A sequence of paths to images used to estimate the
            threshold.  These images should be representative of the
            content you expect to watermark and attack.
        attack_name: Name of the attack to use when generating examples.
            See :func:`attacks` for valid names.  If multiple attacks
            should be applied, you can combine them externally.
        attack_params: Parameters for the chosen attack.  For multiple
            attacks supply a sequence of sequences matching the names.
        fpr_limit: Maximum acceptable false positive rate when selecting
            the threshold.  Used to report a threshold consistent with
            this limit.
        n_samples: Number of attack iterations per image.  Each iteration
            produces one true positive and one true negative score.
        plot_file: Filename for saving the ROC plot.  The file will be
            created in the current working directory.

    Returns:
        The estimated threshold τ on the similarity measure.  The ROC
        curve and AUC are also saved to ``plot_file``.
    """
    # Load the watermark bits once and map them to 0/1
    watermark = np.load(watermark_path).astype(np.uint8).flatten()
    if watermark.size != 1024:
        raise ValueError("Watermark must have length 1024")

    scores: List[float] = []
    labels: List[int] = []

    for img_path in image_paths:
        # Read original image
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        # Embed the watermark once for each original image
        wm_img = embedding(img_path, watermark_path)

        # Precompute DCT of the original and watermarked images
        dct_orig = cv2.dct(orig_img.astype(np.float32))
        dct_wm = cv2.dct(wm_img.astype(np.float32))
        flat = dct_orig.ravel()
        sorted_idx = np.argsort(np.abs(flat))[::-1]
        embed_idx = sorted_idx[1:1025]
        ref_bits = np.zeros(1024, dtype=np.uint8)
        for i, idx in enumerate(embed_idx):
            r, c = divmod(int(idx), dct_orig.shape[1])
            ref_bits[i] = 1 if (dct_wm[r, c] - dct_orig[r, c]) >= 0 else 0

        # For each sample, attack and compute scores
        for _ in range(n_samples):
            # Save watermarked image to temporary file for attacks() API
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, wm_img)
            try:
                attacked = apply_attacks(temp_path, attack_name, attack_params)
            finally:
                os.remove(temp_path)
            # Compute DCT of attacked image
            dct_att = cv2.dct(attacked.astype(np.float32))
            att_bits = np.zeros(1024, dtype=np.uint8)
            for i, idx in enumerate(embed_idx):
                r, c = divmod(int(idx), dct_orig.shape[1])
                att_bits[i] = 1 if (dct_att[r, c] - dct_orig[r, c]) >= 0 else 0
            # True positive score
            tp_score = float(np.sum(att_bits == ref_bits)) / len(ref_bits)
            scores.append(tp_score)
            labels.append(1)
            # True negative: compare with random watermark
            rand_bits = np.random.randint(0, 2, 1024, dtype=np.uint8)
            tn_score = float(np.sum(att_bits == rand_bits)) / len(rand_bits)
            scores.append(tn_score)
            labels.append(0)

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for watermark detection')
    plt.legend(loc="lower right")
    # Save the plot to file
    plt.savefig(plot_file)
    plt.close()

    # Identify thresholds whose FPR is within the desired limit
    valid = thresholds[fpr <= fpr_limit]
    if valid.size == 0:
        tau = float(thresholds[np.argmin(fpr)])
    else:
        tau = float(np.max(valid))

    # Return the threshold value
    return tau


def plot_roc_curve(labels: List[int], scores: List[float]) -> None:
    """Plot the ROC curve given labels and scores.

    Args:
        labels: List of ground truth labels (1 for positive, 0 for negative).
        scores: List of similarity scores corresponding to the labels.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.show()
    