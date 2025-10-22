"""
detection_code.py
===================

This module implements detection for the DCT‑based watermarking scheme
defined in `embedding_code.py`.  Given an original image and a suspect
image (which may or may not contain the watermark), it extracts
embedded values from the suspect image, computes the correlation with
the pseudo‑random watermark sequence generated from the provided key,
and reports a similarity score.  A decision threshold can be provided
on the command line; similarity scores above the threshold indicate
the presence of the watermark.

Usage example:

    python detection_code.py --original path/to/original.png \
        --suspect path/to/watermarked.png --key 12345 --alpha 0.05 \
        --coeffs 2 --channel Cb --threshold 0.5

Parameters
----------
original : str
    Path to the original (unwatermarked) image.
suspect : str
    Path to the image under test.
key : int
    Seed used to generate the watermark sequence during embedding.
alpha : float
    Embedding strength used during embedding.  It is needed for proper
    extraction of the bit estimates.
num_coeffs : int
    Number of DCT coefficients per block used during embedding.
channel : {'Y','Cb','Cr'}
    Colour channel used during embedding.
threshold : float
    Decision threshold for declaring a watermark present.  The
    similarity score lies in the range [-1,1].  A typical threshold is
    between 0.4 and 0.7, selected via ROC analysis.

Returns
-------
It prints the similarity score and whether the watermark is detected
based on the threshold.
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def generate_bipolar_watermark(length: int, key: int) -> np.ndarray:
    """Generate a bipolar watermark sequence identical to the one used in embedding.

    Parameters
    ----------
    length : int
        Number of watermark bits.
    key : int
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of values -1 and +1.
    """
    rng = np.random.default_rng(key)
    bits = rng.integers(0, 2, size=length, dtype=np.int8)
    return (bits * 2 - 1).astype(np.float32)


def extract_block_bits(orig_block: np.ndarray, suspect_block: np.ndarray,
                       alpha: float, num_coeffs: int) -> np.ndarray:
    """Extract estimated watermark bits from a single 8×8 block.

    For multiplicative embedding C' = C * (1 + α m), the estimated bit
    is computed as (C'/C - 1) / α.  The returned estimates are real
    values; correlation across many coefficients is used to detect the
    presence of the watermark.

    Parameters
    ----------
    orig_block : np.ndarray
        Original 8×8 block (float32).
    suspect_block : np.ndarray
        Suspect 8×8 block (float32).
    alpha : float
        Embedding strength used during embedding.
    num_coeffs : int
        Number of coefficients per block to extract.

    Returns
    -------
    np.ndarray
        Array of estimated bits for this block of length `num_coeffs`.
    """
    # Compute DCT for both blocks
    dct_orig = cv2.dct(orig_block)
    dct_susp = cv2.dct(suspect_block)
    positions = [
        (2, 1), (1, 2), (2, 2), (3, 1), (1, 3),
        (3, 2), (2, 3), (4, 1), (1, 4), (4, 2),
        (2, 4), (3, 3)
    ]
    est = np.zeros(num_coeffs, dtype=np.float32)
    for idx in range(num_coeffs):
        pos = positions[idx]
        C = dct_orig[pos]
        C_p = dct_susp[pos]
        # Avoid division by zero or extremely small values
        if abs(C) < 1e-5:
            est[idx] = 0.0
        else:
            est[idx] = (C_p / C - 1.0) / alpha
    return est


def extract_watermark(original_path: str, suspect_path: str, key: int,
                      alpha: float, num_coeffs: int, channel: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract the watermark and compute similarity.

    Parameters
    ----------
    original_path : str
        Path to the original image.
    suspect_path : str
        Path to the image under test.
    key : int
        Seed for watermark generation.
    alpha : float
        Embedding strength used during embedding.
    num_coeffs : int
        Number of DCT coefficients per block used during embedding.
    channel : str
        Colour channel where watermark was embedded.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Extracted estimates, the original watermark sequence, and the
        similarity score (cosine similarity).
    """
    if channel not in {"Y", "Cb", "Cr"}:
        raise ValueError("channel must be one of 'Y', 'Cb', or 'Cr'")
    # Read images in colour (BGR) or grayscale depending on the file
    # We first attempt to load the images as colour; if that fails or the
    # resulting array is two‑dimensional, we treat them as grayscale.
    orig = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    susp = cv2.imread(suspect_path, cv2.IMREAD_UNCHANGED)
    if orig is None or susp is None:
        raise FileNotFoundError("Could not read one of the images.")
    # Determine if images are grayscale (2D) or colour (3D)
    if len(orig.shape) == 2 or len(susp.shape) == 2:
        # For black and white images, work directly on the single channel.
        orig_ch = orig.astype(np.float32)
        susp_ch = susp.astype(np.float32)
    else:
        # Convert to YCrCb and select the desired channel
        orig_ycrcb = cv2.cvtColor(orig, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        susp_ycrcb = cv2.cvtColor(susp, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        channels = {"Y": 0, "Cr": 1, "Cb": 2}
        idx = channels[channel]
        orig_ch = orig_ycrcb[:, :, idx]
        susp_ch = susp_ycrcb[:, :, idx]
    h, w = orig_ch.shape
    block_size = 8
    h_blocks = h // block_size
    w_blocks = w // block_size
    total_blocks = h_blocks * w_blocks
    total_positions = total_blocks * num_coeffs
    # Generate original watermark sequence
    watermark = generate_bipolar_watermark(total_positions, key)
    extracted = np.zeros(total_positions, dtype=np.float32)
    est_idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            orig_block = orig_ch[i * block_size:(i + 1) * block_size,
                                j * block_size:(j + 1) * block_size]
            susp_block = susp_ch[i * block_size:(i + 1) * block_size,
                                j * block_size:(j + 1) * block_size]
            est = extract_block_bits(orig_block, susp_block, alpha, num_coeffs)
            extracted[est_idx:est_idx + num_coeffs] = est
            est_idx += num_coeffs
    # Compute cosine similarity
    # Avoid division by zero
    norm_extracted = np.linalg.norm(extracted)
    norm_wm = np.linalg.norm(watermark)
    if norm_extracted == 0 or norm_wm == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(extracted, watermark) / (norm_extracted * norm_wm))
    return extracted, watermark, similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a DCT‑based watermark in an image.")
    parser.add_argument("--original", required=True, help="Path to the original (unwatermarked) image.")
    parser.add_argument("--suspect", required=True, help="Path to the suspect image.")
    parser.add_argument("--key", type=int, default=12345, help="Seed used during embedding.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Embedding strength used during embedding.")
    parser.add_argument("--coeffs", type=int, default=2, help="Number of DCT coefficients per block.")
    parser.add_argument(
        "--channel",
        choices=["Y", "Cb", "Cr"],
        default="Cb",
        help=(
            "Colour channel used during embedding (ignored for grayscale images). "
            "For black‑and‑white images only the luminance channel is present."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for similarity score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extracted, watermark, similarity = extract_watermark(
        args.original, args.suspect, args.key, args.alpha, args.coeffs, args.channel
    )
    print(f"Similarity score: {similarity:.4f}")
    if similarity >= args.threshold:
        print("Watermark detected (similarity >= threshold).")
    else:
        print("Watermark NOT detected (similarity < threshold).")


if __name__ == "__main__":
    main()