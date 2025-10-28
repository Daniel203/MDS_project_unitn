import os

import cv2
import numpy as np

from attacks import attacks
from detection_ACME import detection
from embedding import embedding
from roc import roc

INPUT_DIR = "input"
"""Folder containing images to use"""

OUTPUT_DIR = "output"
"""Folder where embedded images will be saved"""

WATERMARK_NAME = "watermark.npy"
"""Name of the watermark file"""


def roc_curve():
    images = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".bmp"):
            image_path = os.path.join(INPUT_DIR, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)

    roc(images, np.load(WATERMARK_NAME))


def test_embedding():
    image_original = "path_to_original_image"
    watermark = "path_to_watermark"

    watermarked_image = embedding(image_original, watermark)


def test_detection():
    image_original = "path_to_original_image"
    image_watermarked = "path_to_watermarked_image"
    image_attacked = "path_to_attacked_image"

    output1, output2 = detection(image_original, image_watermarked, image_attacked)

    print(f"Watermark found: {'yes' if output1 == 1 else 'no'}\n")
    print(f"WPSNR: {output2}")


def test_attacks():
    original_image_path = "path_to_original_image"
    watermarked_image_path = "watermarked.bmp"
    attacked_image_path = "attacked.bmp"
    image_watermarked = embedding(original_image_path, WATERMARK_NAME)
    if image_watermarked is None:
        return

    cv2.imwrite(watermarked_image_path, image_watermarked)

    cv2.imwrite(attacked_image_path, image_watermarked)
    attacked_image = attacks(attacked_image_path, "median", [5])
    cv2.imwrite(attacked_image_path, attacked_image)

    cv2.imwrite(attacked_image_path, attacked_image)
    detect1, detect2 = detection(
        original_image_path, watermarked_image_path, attacked_image_path
    )

    print(f"watermark detected: {'yes' if detect1 == 1 else 'no'}")
    print(f"WPSNR attacked image: {detect2}")


if __name__ == "__main__":
    roc_curve()
    test_attacks()
    test_embedding()
    test_detection()
