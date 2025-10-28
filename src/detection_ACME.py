import cv2
import numpy as np
import pywt
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.fft import dct

from wpsnr import wpsnr

MARK_SIZE = 1024
"""The size of the watermark. It's 1024 as defined in the challange constraints"""

ALPHA = 1.0
"""The alpha value used when embedding and decoding"""

THRESH = 0.54
"""Threshold value used to determine if a detected watermark is valid or not"""

MID_FREQ_START = 5000
"""Skip the first MID_FREQ_START frequencies when embedding and detecting"""



def detection(input1: str, input2: str, input3: str) -> tuple[int, float]:
    image_original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    image_watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    image_attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    if image_original is None:
        raise ValueError(f"Failed to load original image from {input1}")
    if image_watermarked is None:
        raise ValueError(f"Failed to load watermarked image from {input2}")
    if image_attacked is None:
        raise ValueError(f"Failed to load attacked image from {input3}")

    params = [ALPHA, "additive"]

    watermark_original = extract_watermark(image_original, image_watermarked, params)
    watermark_attacked = extract_watermark(image_original, image_attacked, params)

    output1 = 1 if similarity(watermark_original, watermark_attacked) > THRESH else 0
    output2 = wpsnr(image_watermarked, image_attacked)

    return output1, output2


def extract_watermark(
    original: MatLike, watermarked: MatLike, params: list
) -> NDArray[np.float64]:
    alpha = params[0]
    strategy = params[1]

    # Get LL sub-band for both images
    LL_ori, (_) = pywt.dwt2(original, "haar")
    LL_wat, (_) = pywt.dwt2(watermarked, "haar")

    # Get dct of both LL sub-bands
    ori_dct = dct(dct(LL_ori, axis=0, norm="ortho"), axis=1, norm="ortho")
    wat_dct = dct(dct(LL_wat, axis=0, norm="ortho"), axis=1, norm="ortho")

    # Find locations
    abs_ori_dct = abs(ori_dct)
    locations = np.argsort(-abs_ori_dct, axis=None)
    rows = LL_ori.shape[0]  # Use LL sub-band shape
    locations = [(val // rows, val % rows) for val in locations]

    # Empty array that will contain the extracted watermark
    w_ex = np.zeros(MARK_SIZE, dtype=np.float64)

    start_index = MID_FREQ_START
    end_index = start_index + MARK_SIZE

    for i, loc in enumerate(locations[start_index:end_index]):
        if strategy == "additive":
            w_ex[i] = (wat_dct[loc] - ori_dct[loc]) / alpha
        elif strategy == "multiplicative":
            if ori_dct[loc] != 0 and alpha != 0:  # Avoid divide by zero
                w_ex[i] = (wat_dct[loc] / ori_dct[loc] - 1) / alpha
            else:
                w_ex[i] = 0

    return w_ex


def similarity(watermark_1, watermarked_2):
    s = np.sum(np.multiply(watermark_1, watermarked_2)) / (
        np.sqrt(np.sum(np.multiply(watermark_1, watermark_1)))
        * np.sqrt(np.sum(np.multiply(watermarked_2, watermarked_2)))
    )

    # Rescales the result from [-1, 1] to [0, 1]
    res = (s + 1) / 2
    return res
