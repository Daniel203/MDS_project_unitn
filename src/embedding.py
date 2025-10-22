"""
Watermark embedding using a spread‑spectrum scheme in the DCT domain.

This module provides a single function, :func:`embedding`, which embeds a
1024‑bit binary watermark into a 512×512 grayscale image.  The embedding
procedure follows the guidelines discussed in the laboratory materials:

* The host image is transformed to the frequency domain using a 2D DCT.
* The magnitudes of the transform coefficients (except the DC term) are
  sorted in descending order and the 1024 largest coefficients are
  selected for embedding.  This selects mid to high frequency content
  which offers a balance between invisibility and robustness.
* Each watermark bit (0 or 1) is mapped to ±1 and applied multiplicatively
  to the selected DCT coefficients.  A small modulation factor ``ALPHA``
  controls the trade‑off between robustness and imperceptibility.  A
  positive watermark bit increases the magnitude of the coefficient, a
  negative bit decreases it.  The DC coefficient is never modified.
* The inverse DCT reconstructs the spatial image.  The result is
  clipped to [0,255] and returned as an 8‑bit array.

The function does not print anything or open any windows.  It is designed
to be called programmatically.  You may adjust the global ``ALPHA``
constant below to fine‑tune the robustness of your watermark.  Lower
values of ``ALPHA`` improve invisibility at the cost of robustness.

Note that this routine assumes the watermark file contains a 1D array
of length 1024 consisting of zeros and ones.  If your watermark has a
different shape you should reshape or flatten it before saving.
"""

from __future__ import annotations
import cv2
import numpy as np

# Modulation strength for the spread‑spectrum embedding.  Values around
# 0.05–0.2 offer a reasonable trade‑off between invisibility and
# robustness.  A smaller value yields less visible artifacts but makes
# detection more sensitive to attacks.
ALPHA: float = 0.1


def embedding(input1: str, input2: str) -> np.ndarray:
    """Embed a binary watermark into a grayscale image using DCT.

    Args:
        input1: Path to the original image on disk.  The image must be
            grayscale (one channel) and of size 512×512.
        input2: Path to the watermark file (``.npy``) on disk containing
            exactly 1024 bits (0 or 1).

    Returns:
        The watermarked image as a NumPy array of dtype ``uint8``.  The
        caller may write this image to disk if needed.
    """
    # Read the original image and ensure it exists.
    host = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if host is None:
        raise FileNotFoundError(f"Could not read input image: {input1}")
    if host.shape[0] != 512 or host.shape[1] != 512:
        raise ValueError("Input image must be 512×512 pixels")

    # Load the watermark bits.  Flatten to 1D and verify length.
    watermark = np.load(input2).astype(np.uint8).flatten()
    if watermark.ndim != 1 or watermark.size != 1024:
        raise ValueError("Watermark must be a one‑dimensional array of length 1024")

    # Map bits from {0,1} to {-1,1}.  Bit 0 becomes ‑1, bit 1 becomes +1.
    wm_mapped = 2 * watermark.astype(np.float32) - 1.0

    # Convert the host image to float32 for DCT.
    host_f = host.astype(np.float32)
    # Compute the 2D Discrete Cosine Transform (DCT).
    dct_host = cv2.dct(host_f)

    # Flatten and sort coefficients by magnitude (descending).  Skip the
    # first element (DC term) which is typically the largest.
    flat = dct_host.ravel()
    # Indices sorted by absolute value descending
    sorted_indices = np.argsort(np.abs(flat))[::-1]
    # Exclude the DC component at index 0 and take the next 1024 indices
    embed_indices = sorted_indices[1:1025]

    # Embed the watermark multiplicatively.
    for i, idx in enumerate(embed_indices):
        r, c = divmod(int(idx), dct_host.shape[1])
        coeff = dct_host[r, c]
        # Apply multiplicative modulation: c' = c * (1 + alpha * w)
        dct_host[r, c] = coeff * (1.0 + ALPHA * wm_mapped[i])

    # Compute the inverse DCT to obtain the watermarked image.
    watermarked_f = cv2.idct(dct_host)
    # Round and clip values to valid intensity range.
    watermarked = np.clip(np.round(watermarked_f), 0, 255).astype(np.uint8)

    return watermarked