import numpy as np
import pywt
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.fft import dct

from constraints import MARK_SIZE, MID_FREQ_START
from embed import EmbeddingStrategy, EmbedParameters


def extract_watermark(
    original: MatLike, watermarked: MatLike, params: EmbedParameters
) -> NDArray[np.float64]:
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
        if params.strategy == EmbeddingStrategy.ADDITIVE:
            w_ex[i] = (wat_dct[loc] - ori_dct[loc]) / params.alpha
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            if ori_dct[loc] != 0 and params.alpha != 0:  # Avoid divide by zero
                # w_ex[i] = (wat_dct[loc] - ori_dct[loc]) / (
                #     params.alpha * ori_dct[loc]
                # )
                w_ex[i] = (wat_dct[loc] / ori_dct[loc] - 1) / params.alpha
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
    # return s
