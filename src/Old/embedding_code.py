"""
embedding_code.py
===================

This module implements a simple spread‑spectrum watermark embedding scheme
based on the Discrete Cosine Transform (DCT).  The watermark is embedded
using multiplicative modulation of mid‑frequency DCT coefficients in
8×8 blocks.  For colour images, the embedding takes place in a
specified channel of the YCbCr colour space (default Cb).  For
grayscale (black‑and‑white) images, the watermark is embedded directly
into the single channel.  The goal is to produce watermarked images
that are robust to common attacks yet remain imperceptible when viewed.

Usage example:

    python embedding_code.py --images path/to/img1.png path/to/img2.png path/to/img3.png \
        --alpha 0.05 --key 12345 --coeffs 2 --channel Cb

This will produce watermarked images with the suffix `_wm` added before the
extension, e.g. `img1_wm.png`.

Parameters
----------
image_path : str
    Path to the input image.  Supported formats are those understood by
    OpenCV (PNG, JPG, BMP, etc.).
watermark_key : int, optional
    Seed for the pseudo‑random number generator used to generate the
    watermark sequence.  Using the same key for embedding and detection
    ensures reproducibility.
alpha : float, optional
    Embedding strength.  Larger values increase robustness but may reduce
    perceptual quality.  Values in the range 0.05–0.10 are typically
    suitable.
num_coeffs : int, optional
    Number of DCT coefficients per 8×8 block used to embed a single
    watermark bit.  A value of 2 means each block embeds two bits.
channel : {'Cb', 'Cr', 'Y'}, optional
    Colour channel in which to embed the watermark.  Embedding in the
    chroma channels (Cb or Cr) is recommended because these channels have
    higher redundancy and modifications are less visible to the human eye.

The script may be extended to embed into both Cb and Cr channels or to
incorporate additional redundancy (e.g. replication across multiple
frequency bands).  Detection code is not included here but can be
implemented using the inverse process described in the project report.
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def generate_bipolar_watermark(length: int, key: int) -> np.ndarray:
    """Generate a bipolar watermark sequence of the given length.

    The sequence contains values -1 and +1.  A reproducible random
    generator seeded by `key` is used to ensure that the same key
    produces the same sequence for both embedding and detection.

    Parameters
    ----------
    length : int
        Desired length of the watermark sequence.
    key : int
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        A 1‑D array of shape (length,) with values in {−1, +1}.
    """
    rng = np.random.default_rng(key)
    # Generate random bits {0, 1}, map to {−1, +1}
    bits = rng.integers(0, 2, size=length, dtype=np.int8)
    return (bits * 2 - 1).astype(np.float32)


def embed_dct_block(block: np.ndarray, wm_bits: np.ndarray, alpha: float) -> None:
    """Embed watermark bits into a single 8×8 block using DCT.

    The function modifies the block in place.  It selects a fixed set of
    mid‑frequency coefficients ((2,1), (1,2), (2,3), (3,2), etc.) based on
    the length of `wm_bits` and applies multiplicative modulation:

        C' = C * (1 + alpha * bit)

    where C is the original DCT coefficient and bit is ±1.

    Parameters
    ----------
    block : np.ndarray
        A single 8×8 block (float32) from the selected colour channel.
    wm_bits : np.ndarray
        A 1‑D array containing the bits to embed into this block.  Its
        length determines how many coefficients are modified.  If fewer
        bits are provided than coefficients, the remaining coefficients
        are left unmodified.
    alpha : float
        Embedding strength.
    """
    # Compute forward DCT
    dct_block = cv2.dct(block)
    # Predefined positions for embedding (mid‑frequency coefficients)
    positions = [
        (2, 1), (1, 2), (2, 2), (3, 1), (1, 3),
        (3, 2), (2, 3), (4, 1), (1, 4), (4, 2),
        (2, 4), (3, 3)
    ]
    # Only modify as many coefficients as there are bits
    for idx, bit in enumerate(wm_bits):
        if idx >= len(positions):
            break
        pos = positions[idx]
        # Multiplicative embedding
        dct_block[pos] *= (1.0 + alpha * bit)
    # Inverse DCT
    inv_block = cv2.idct(dct_block)
    block[:, :] = inv_block


def embed_watermark_into_channel(channel: np.ndarray, watermark: np.ndarray, alpha: float, num_coeffs: int) -> np.ndarray:
    """Embed a watermark sequence into a single colour channel.

    The channel is divided into non‑overlapping 8×8 blocks.  Each block
    embeds `num_coeffs` bits from the watermark sequence by modifying
    selected DCT coefficients.  If the watermark is shorter than the
    number of available embedding positions, the sequence repeats.

    Parameters
    ----------
    channel : np.ndarray
        2‑D image array (float32) of the selected colour channel.
    watermark : np.ndarray
        1‑D array of bipolar bits to embed.
    alpha : float
        Embedding strength.
    num_coeffs : int
        Number of coefficients per block to embed.  Must be ≤ 12, the
        number of predefined embedding positions.

    Returns
    -------
    np.ndarray
        The channel with the watermark embedded.
    """
    h, w = channel.shape
    block_size = 8
    h_blocks = h // block_size
    w_blocks = w // block_size
    total_blocks = h_blocks * w_blocks
    # Compute total embedding positions
    total_positions = total_blocks * num_coeffs
    if len(watermark) < total_positions:
        # Repeat watermark to fill all positions
        repeats = int(np.ceil(total_positions / len(watermark)))
        watermark = np.tile(watermark, repeats)[:total_positions]
    wm_idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = channel[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bits_for_block = watermark[wm_idx:wm_idx + num_coeffs]
            embed_dct_block(block, bits_for_block, alpha)
            wm_idx += num_coeffs
    return channel


def embed_watermark(
    image_path: str,
    key: int = 12345,
    alpha: float = 0.05,
    num_coeffs: int = 2,
    channel: str = "Cb",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed a watermark into an image.

    This function supports both colour and grayscale images.  For colour
    images, the watermark is embedded into the specified channel of the
    YCbCr colour space.  For grayscale (single‑channel) images, the
    watermark is embedded directly into the image's only channel.  The
    embedding uses multiplicative modulation of mid‑frequency DCT
    coefficients in 8×8 blocks, as described in the project report.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    key : int, optional
        Seed for the watermark generator.
    alpha : float, optional
        Embedding strength.
    num_coeffs : int, optional
        Number of DCT coefficients per block to embed.  Must be ≤ 12.
    channel : str, optional
        Which channel to embed into for colour images: 'Y', 'Cb', or
        'Cr'.  Embedding into 'Cb' or 'Cr' is recommended because these
        channels have higher redundancy and modifications are less
        visible【651565660099664†L0-L8】.  This parameter is ignored for
        grayscale images.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the watermarked image (uint8, with the same
        number of channels as the input), the generated bipolar
        watermark sequence, and the modified YCbCr/gray image (float32).
    """
    # Load image in unchanged mode to preserve number of channels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # Determine if image is grayscale (2D) or colour (3D)
    if len(img.shape) == 2:
        # Grayscale: embed directly in the single channel
        gray = img.astype(np.float32)
        h, w = gray.shape
        block_size = 8
        total_blocks = (h // block_size) * (w // block_size)
        total_positions = total_blocks * num_coeffs
        watermark = generate_bipolar_watermark(total_positions, key)
        embedded = embed_watermark_into_channel(gray.copy(), watermark, alpha, num_coeffs)
        # Clip and convert back to uint8
        watermarked = np.clip(embedded, 0, 255).astype(np.uint8)
        return watermarked, watermark, embedded
    else:
        # Colour image: embed into specified YCbCr channel
        if channel not in {"Y", "Cb", "Cr"}:
            raise ValueError("channel must be one of 'Y', 'Cb', or 'Cr'")
        # Convert BGR to YCrCb (note: OpenCV uses YCrCb ordering)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y, Cr, Cb = cv2.split(ycrcb)
        # Select channel for embedding
        if channel == "Y":
            target = Y
        elif channel == "Cb":
            target = Cb
        else:  # "Cr"
            target = Cr
        # Generate watermark sequence
        h, w = target.shape
        block_size = 8
        total_blocks = (h // block_size) * (w // block_size)
        total_positions = total_blocks * num_coeffs
        watermark = generate_bipolar_watermark(total_positions, key)
        # Embed watermark
        embedded_channel = embed_watermark_into_channel(target.copy(), watermark, alpha, num_coeffs)
        # Reconstruct channels
        if channel == "Y":
            Y = embedded_channel
        elif channel == "Cb":
            Cb = embedded_channel
        else:
            Cr = embedded_channel
        watermarked_ycrcb = cv2.merge([Y, Cr, Cb])
        # Convert back to BGR
        watermarked_bgr = cv2.cvtColor(watermarked_ycrcb.astype(np.float32), cv2.COLOR_YCrCb2BGR)
        watermarked_bgr = np.clip(watermarked_bgr, 0, 255).astype(img.dtype)
        return watermarked_bgr, watermark, watermarked_ycrcb


def save_watermarked_image(image: np.ndarray, original_path: str) -> str:
    """Save a watermarked image next to the original with a `_wm` suffix.

    Parameters
    ----------
    image : np.ndarray
        Watermarked image array.  It may be grayscale (2‑D) or colour
        (3‑D).  The function writes it using OpenCV without further
        conversion.
    original_path : str
        Path to the original image.  The output name is derived from
        this path by inserting `_wm` before the file extension.

    Returns
    -------
    str
        The path to the saved watermarked image.
    """
    base, ext = os.path.splitext(original_path)
    out_path = f"{base}_wm{ext}"
    cv2.imwrite(out_path, image)
    return out_path


def process_images(image_paths: List[str], key: int, alpha: float, num_coeffs: int, channel: str) -> None:
    """Embed watermarks into multiple images and save the results.

    This helper function iterates over `image_paths`, embeds a watermark
    using the specified parameters, and writes the watermarked images
    with the `_wm` suffix.  It prints the output paths for reference.

    Parameters
    ----------
    image_paths : List[str]
        List of image file paths to process.
    key : int
        Seed for the watermark generator.
    alpha : float
        Embedding strength.
    num_coeffs : int
        Number of DCT coefficients per block used for embedding.
    channel : str
        Colour channel to embed into.
    """
    for img_path in image_paths:
        try:
            wm_img, wm_seq, _ = embed_watermark(img_path, key=key, alpha=alpha,
                                               num_coeffs=num_coeffs, channel=channel)
            out = save_watermarked_image(wm_img, img_path)
            print(f"Embedded watermark into {img_path}; saved to {out}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Embed a DCT‑based watermark into one or more images.")
    parser.add_argument("--images", default=["images\\0100.bmp"],  nargs="+", help="Paths to images to watermark.")
    parser.add_argument("--key", type=int, default=12345, help="Seed for watermark generation.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Embedding strength.")
    parser.add_argument("--coeffs", type=int, default=2, help="Number of DCT coefficients per block to embed.")
    parser.add_argument(
        "--channel",
        choices=["Y", "Cb", "Cr"],
        default="Cb",
        help=(
            "Colour channel for embedding in colour images (ignored for grayscale)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_images(args.images, args.key, args.alpha, args.coeffs, args.channel)


if __name__ == "__main__":
    main()