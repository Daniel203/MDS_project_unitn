import numpy as np
import os
import cv2
from tempfile import TemporaryDirectory


def jpeg_compression(image: np.ndarray, quality_factor: int):
    with TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "temp.jpg")
        cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        attacked = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return attacked
