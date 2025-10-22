"""Example usage of the spread‑spectrum watermarking toolkit.

This script demonstrates how to call the embedding, attack and
detection functions implemented for the multimedia data security
competition.  It operates on 512×512 grayscale images using the
spread‑spectrum DCT method and the permitted image processing
attacks.  The example uses the classic ``lena_grey.bmp`` image and
``mark.npy`` watermark provided in the laboratory materials.

When executed, the script will:

1. Embed the watermark into the original image.
2. Apply a single attack (additive white Gaussian noise) to the
   watermarked image.
3. Attempt to detect the watermark in the attacked image, printing
   the binary result and the WPSNR.
4. Compute an example similarity threshold using a minimal set of
   images.  For meaningful ROC analysis you should supply many
   representative images and random attack parameters.

For competition submission you should call the individual functions
directly rather than relying on this script.  Ensure that all sample
files reside in the current directory or adjust the paths below.
"""

from __future__ import annotations
import os
import numpy as np
import cv2
from embedding import embedding
from attacks import attacks
from detection_groupname import detection
from roc_threshold import compute_threshold


def main() -> None:
    # Example files.  Update these paths to point to your own images.
    original_image = 'images\\0100.bmp'
    watermark_file = 'src\\acme.npy'

    if not os.path.isfile(original_image):
        raise FileNotFoundError(f"Cannot find example image: {original_image}")
    if not os.path.isfile(watermark_file):
        raise FileNotFoundError(f"Cannot find watermark file: {watermark_file}")

    # Step 1: Embed the watermark
    watermarked = embedding(original_image, watermark_file)
    watermarked_path = 'demo_watermarked.bmp'
    cv2.imwrite(watermarked_path, watermarked)
    print(f"Watermarked image saved to {watermarked_path}")

    # Step 2: Perform an attack (AWGN in this example)
    # attacked = attacks(watermarked_path, 'awgn', [10])
    attacked = attacks(watermarked_path, 'awgn', 10)


    attacked_path = 'demo_attacked.bmp'
    cv2.imwrite(attacked_path, attacked)
    print(f"Attacked image saved to {attacked_path}")

    # Step 3: Detect the watermark
    result, wpsnr_value = detection(original_image, watermarked_path, attacked_path)
    print(f"Detection result (1=present, 0=absent): {result}")
    print(f"WPSNR between watermarked and attacked images: {wpsnr_value:.2f} dB")

    # Step 4: Compute a threshold using the ROC procedure on two
    # images.  In practice you should provide more images for better
    # estimation.
    tau = compute_threshold(watermark_file, [original_image], attack_name='awgn', attack_params=[10], fpr_limit=0.1)
    print(f"Estimated similarity threshold (τ) for FPR ≤ 0.1: {tau:.4f}")


if __name__ == '__main__':
    main()