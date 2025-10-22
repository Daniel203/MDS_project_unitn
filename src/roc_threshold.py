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
from sklearn.metrics import roc_curve
from embedding import embedding
from attacks import attacks as apply_attacks
from attacks import randomized_attack
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


# -----------------------------------------------------------------------------
# New functionality for ROC analysis with randomized attacks


def compute_roc_curve_random(
    watermark_path: str,
    image_path: str,
    n_samples: int = 500,
    fpr_limit: float = 0.1,
    plot_file: str | None = None,
) -> float:
    """Compute a ROC curve using randomised attacks for a single image.

    This function mirrors the procedure demonstrated in the LAB4‑ROC
    notebook: a watermark is embedded into the given image, and then a
    series of random attacks are applied to the watermarked image.  For
    each attacked image we extract the watermark bits relative to the
    original image and compute a similarity measure.  A corresponding
    similarity is also computed against a random watermark to estimate
    false positives.  The resulting scores and labels are used to
    compute a ROC curve and select a threshold τ such that the false
    positive rate does not exceed ``fpr_limit``.

    Args:
        watermark_path: Path to the ``.npy`` file containing the 1024‑bit
            watermark.
        image_path: Path to the original host image (512×512 grayscale).
        n_samples: Number of random attack samples to generate.  Each
            sample yields one true positive and one false positive
            similarity score.
        fpr_limit: Maximum acceptable false positive rate when
            selecting the threshold.
        plot_file: Optional filename for saving the ROC curve plot.  If
            provided, the figure will be written to this file instead
            of being displayed on screen.  When ``None``, the plot is
            shown interactively.

    Returns:
        The selected threshold τ.  Use this value as the detection
        threshold in ``detection_groupname.py``.
    """
    # Load watermark bits and verify length
    watermark = np.load(watermark_path).astype(np.uint8).flatten()
    if watermark.size != 1024:
        raise ValueError("Watermark must have length 1024")

    # Load original image
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if orig.shape[0] != 512 or orig.shape[1] != 512:
        raise ValueError("Input image must be 512×512 pixels")

    # Embed watermark once
    watermarked = embedding(image_path, watermark_path)

    # Precompute the indices of the embedding positions using the original
    f_orig = orig.astype(np.float32)
    dct_orig = cv2.dct(f_orig)
    flat = dct_orig.ravel()
    sorted_idx = np.argsort(np.abs(flat))[::-1]
    embed_indices = sorted_idx[1:1025]

    # Precompute the reference sign bits from the watermarked image
    f_wm = watermarked.astype(np.float32)
    dct_wm = cv2.dct(f_wm)
    ref_sign = np.zeros(1024, dtype=np.int8)
    for i, idx in enumerate(embed_indices):
        r, c = divmod(int(idx), dct_orig.shape[1])
        # Map >=0 difference to +1, else -1
        ref_sign[i] = 1 if (dct_wm[r, c] - dct_orig[r, c]) >= 0 else -1

    # Lists to accumulate similarity scores and labels
    scores: List[float] = []
    labels: List[int] = []

    # Iterate over random attacks
    sample = 0
    while sample < int(n_samples):
        # Generate a random 1024‑bit watermark for H0 (false positive)
        fake_bits = np.random.randint(0, 2, 1024, dtype=np.uint8)
        fake_sign = 2 * fake_bits.astype(np.int8) - 1

        # Apply a random attack to the watermarked image (as an array)
        attacked = randomized_attack(watermarked)
        f_att = attacked.astype(np.float32)
        dct_att = cv2.dct(f_att)

        # Extract sign bits from the attacked image relative to the original
        att_sign = np.zeros(1024, dtype=np.int8)
        for i, idx in enumerate(embed_indices):
            r, c = divmod(int(idx), dct_orig.shape[1])
            att_sign[i] = 1 if (dct_att[r, c] - dct_orig[r, c]) >= 0 else -1

        # Similarity between reference watermark and attacked watermark (H1)
        sim_tp = float(np.sum(ref_sign == att_sign)) / len(ref_sign)
        scores.append(sim_tp)
        labels.append(1)

        # Similarity between fake watermark and attacked watermark (H0)
        sim_fp = float(np.sum(fake_sign == att_sign)) / len(fake_sign)
        scores.append(sim_fp)
        labels.append(0)

        sample += 1

    # Compute ROC curve and threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Select the maximum threshold with FPR <= fpr_limit
    valid_thresholds = thresholds[fpr <= fpr_limit]
    if valid_thresholds.size == 0:
        tau = float(thresholds[np.argmin(fpr)])
    else:
        tau = float(np.max(valid_thresholds))
    # Clip tau to the valid similarity range [0, 1].  ROC thresholds
    # returned by ``roc_curve`` can sometimes exceed the score
    # range when the underlying score distribution is degenerate.
    tau = max(0.0, min(1.0, tau))

    # Plot ROC curve
    if plot_file:
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()
    else:
        # Display interactively
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.show()

    return tau


def compute_roc_curve_from_dir(
    watermark_path: str,
    images_dir: str,
    n_samples: int = 500,
    fpr_limit: float = 0.1,
    plot_file: str | None = None,
) -> float:
    """Estimate a ROC curve using all images in a directory.

    This convenience function embeds the watermark into every image
    found in ``images_dir`` (non‑recursively) and then generates
    ``n_samples`` randomised attack samples per image.  Similarity
    scores are accumulated across all images and used to compute a
    single ROC curve.  The threshold τ is selected such that the
    overall false positive rate does not exceed ``fpr_limit``.

    Args:
        watermark_path: Path to the ``.npy`` watermark file (length 1024).
        images_dir: Directory containing 512×512 grayscale images.  Any
            file ending in ``.bmp``, ``.png`` or ``.jpg`` will be processed.
        n_samples: Number of random attack iterations per image.
        fpr_limit: Maximum false positive rate for selecting τ.
        plot_file: Optional filename to save the ROC curve plot.  When
            ``None`` the plot is displayed interactively.

    Returns:
        The threshold τ derived from the combined ROC curve.
    """
    # Validate watermark
    watermark = np.load(watermark_path).astype(np.uint8).flatten()
    if watermark.size != 1024:
        raise ValueError("Watermark must have length 1024")

    # Collect image paths with supported extensions
    supported_exts = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = []
    for fname in sorted(os.listdir(images_dir)):
        if any(fname.lower().endswith(ext) for ext in supported_exts):
            image_files.append(os.path.join(images_dir, fname))
    if not image_files:
        raise FileNotFoundError(f"No supported images found in directory: {images_dir}")

    scores: List[float] = []
    labels: List[int] = []

    # Process each image
    for img_path in image_files:
        # Load original image and verify size
        orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            # Skip unreadable files
            continue
        if orig.shape[0] != 512 or orig.shape[1] != 512:
            # Skip images with unexpected dimensions
            continue
        # Embed watermark
        watermarked = embedding(img_path, watermark_path)
        # Precompute DCT and embedding indices
        f_orig = orig.astype(np.float32)
        dct_orig = cv2.dct(f_orig)
        flat = dct_orig.ravel()
        sorted_idx = np.argsort(np.abs(flat))[::-1]
        embed_indices = sorted_idx[1:1025]
        # Precompute reference sign bits
        f_wm = watermarked.astype(np.float32)
        dct_wm = cv2.dct(f_wm)
        ref_sign = np.zeros(1024, dtype=np.int8)
        for i, idx in enumerate(embed_indices):
            r, c = divmod(int(idx), dct_orig.shape[1])
            ref_sign[i] = 1 if (dct_wm[r, c] - dct_orig[r, c]) >= 0 else -1
        # Generate n_samples random attacks for this image
        sample = 0
        while sample < int(n_samples):
            # Fake watermark for H0
            fake_bits = np.random.randint(0, 2, 1024, dtype=np.uint8)
            fake_sign = 2 * fake_bits.astype(np.int8) - 1
            # Attack watermarked image using randomised attack
            attacked = randomized_attack(watermarked)
            f_att = attacked.astype(np.float32)
            dct_att = cv2.dct(f_att)
            # Extract attacked sign bits
            att_sign = np.zeros(1024, dtype=np.int8)
            for i, idx in enumerate(embed_indices):
                r, c = divmod(int(idx), dct_orig.shape[1])
                att_sign[i] = 1 if (dct_att[r, c] - dct_orig[r, c]) >= 0 else -1
            # H1 similarity
            sim_tp = float(np.sum(ref_sign == att_sign)) / len(ref_sign)
            scores.append(sim_tp)
            labels.append(1)
            # H0 similarity
            sim_fp = float(np.sum(fake_sign == att_sign)) / len(fake_sign)
            scores.append(sim_fp)
            labels.append(0)
            sample += 1

    # Compute ROC and threshold from aggregated scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid_thresholds = thresholds[fpr <= fpr_limit]
    if valid_thresholds.size == 0:
        tau = float(thresholds[np.argmin(fpr)])
    else:
        tau = float(np.max(valid_thresholds))
    # Clip tau to [0, 1] as similarity scores lie in this interval
    tau = max(0.0, min(1.0, tau))

    # Plot aggregated ROC
    if plot_file:
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (all images)')
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()
    else:
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (all images)')
        plt.grid(True)
        plt.show()

    return tau
    