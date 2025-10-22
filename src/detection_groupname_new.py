"""
Watermark detection script for the multimedia competition.

This module implements a non‑blind detector for the spread‑spectrum
watermark embedded by :func:`embedding` in the companion module.  It
follows the recommendations from the laboratory notes:

* The original image, watermarked image and attacked image are all
  required inputs.  Detection is therefore non‑blind, which is
  encouraged for robustness.
* Detection is performed in the DCT domain.  The same set of 1024
  highest‑magnitude DCT coefficients (excluding the DC term) used for
  embedding are recovered by sorting the coefficients of the original
  image.  We then derive the sign of the watermark modulation from
  differences between the watermarked and original DCT coefficients.
  Comparing those signs with the corresponding signs from the attacked
  image yields a similarity measure in [0,1].
* A decision threshold ``TAU`` is applied to the similarity measure to
  decide whether the watermark is present.  Users should estimate a
  suitable value for ``TAU`` off‑line using ROC analysis (see
  ``roc_threshold.py``) and replace the default below.
* The function also returns the weighted PSNR (WPSNR) between the
  watermarked and attacked images.  This value should exceed 35 dB to
  meet the competition quality constraint.

The WPSNR implementation below reproduces the lab code (with minor
formatting changes) and requires SciPy for convolution.  Do not print
anything or open any windows from this script.  It is designed to be
called automatically during the competition.
"""

from typing import Tuple
import numpy as np
import cv2

# Import the embedding module to access the modulation strength.  The
# ``ALPHA`` constant controls how strongly each watermark bit modulates
# the DCT coefficients during embedding.  We reuse the same value here
# during detection to properly reverse the embedding operation.  See
# ``embedding.py`` for the definition of ``ALPHA``.
from embedding import ALPHA  # type: ignore

# Import the official WPSNR implementation provided by the professor.  This
# module defines a function ``wpsnr`` that computes the weighted PSNR
# between two images.  Using this module satisfies the requirement to
# employ the supplied WPSNR implementation rather than re‑implementing
# the function here.
from wpsnr import wpsnr


# -----------------------------------------------------------------------------
# Previously this file contained a local implementation of the WPSNR
# calculation.  At the user's request we now import the official
# implementation from ``wpsnr.py``.  The helper functions (_csffun,
# _csfmat, _get_csf_filter) are no longer needed and have been removed.




# -----------------------------------------------------------------------------
# Watermark extraction and similarity

# Detection threshold.  Estimate this using the ``compute_threshold``
# function in roc_threshold.py and replace the default below.  A value
# around 0.5–0.7 is often appropriate for spread‑spectrum schemes.
TAU: float = 0.5


def detection(input1: str, input2: str, input3: str) -> Tuple[int, float]:
    """Detect a spread‑spectrum watermark and compute the WPSNR.

    Args:
        input1: Path to the original (unwatermarked) image.
        input2: Path to the watermarked image.
        input3: Path to the attacked image.

    Returns:
        A tuple ``(output1, output2)`` where ``output1`` is ``1`` if
        the watermark is deemed present (similarity ≥ ``TAU``) and
        ``0`` otherwise, and ``output2`` is the WPSNR between the
        watermarked and attacked images in decibels.
    """
    # Load the three images in grayscale
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if original is None or watermarked is None or attacked is None:
        raise FileNotFoundError("Could not read one of the input images")
    # Convert to float32 for DCT
    f_original = original.astype(np.float32)
    f_watermarked = watermarked.astype(np.float32)
    f_attacked = attacked.astype(np.float32)
    # Compute 2D DCTs
    dct_orig = cv2.dct(f_original)
    dct_wm = cv2.dct(f_watermarked)
    dct_att = cv2.dct(f_attacked)
    # Sort DCT coefficients of the original image by magnitude
    flat = dct_orig.ravel()
    sorted_idx = np.argsort(np.abs(flat))[::-1]
    # Exclude DC term and select the next 1024 coefficients
    embed_indices = sorted_idx[1:1025]
    # Derive watermark bits from the relative change in DCT coefficients.
    # Following the laboratory notes, we recover each bit by dividing the
    # difference between the watermarked (or attacked) and original
    # coefficients by ``ALPHA`` times the original coefficient.  This
    # reverses the multiplicative embedding c' = c * (1 + alpha * w).
    ref_bits = np.zeros(1024, dtype=np.uint8)
    att_bits = np.zeros(1024, dtype=np.uint8)
    for i, idx in enumerate(embed_indices):
        r, c = divmod(int(idx), dct_orig.shape[1])
        # Avoid division by zero if the original coefficient is zero
        orig_coeff = dct_orig[r, c]
        if orig_coeff == 0:
            # If the original coefficient is zero, fall back to sign comparison
            ref_val = dct_wm[r, c] - dct_orig[r, c]
            att_val = dct_att[r, c] - dct_orig[r, c]
        else:
            ref_val = (dct_wm[r, c] - dct_orig[r, c]) / (ALPHA * orig_coeff)
            att_val = (dct_att[r, c] - dct_orig[r, c]) / (ALPHA * orig_coeff)
        # For multiplicative embedding, a positive value corresponds to bit=1
        # and a negative value corresponds to bit=0.  Zero values are mapped
        # to bit=1 by convention to bias towards detection.
        ref_bits[i] = 1 if ref_val >= 0 else 0
        att_bits[i] = 1 if att_val >= 0 else 0
    # Compute similarity as fraction of matching bits
    similarity = float(np.sum(ref_bits == att_bits)) / len(ref_bits)
    # Compute WPSNR between the watermarked and attacked images
    wpsnr_value = wpsnr(watermarked, attacked)
    # Make the binary decision based on threshold TAU
    output1 = 1 if similarity >= TAU else 0
    return output1, float(wpsnr_value)