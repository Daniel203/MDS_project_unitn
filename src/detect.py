import numpy as np
from cv2.typing import MatLike
from scipy.fft import dct

from constraints import MARK_SIZE, MID_FREQ_START
from embed import EmbeddingStrategy, EmbedParameters


def extract_watermark(image: MatLike, watermarked: MatLike, params: EmbedParameters):
    ori_dct = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
    wat_dct = dct(dct(watermarked, axis=0, norm="ortho"), axis=1, norm="ortho")

    # Get the locations of the most perceptually significant components
    abs_ori_dct = abs(ori_dct)
    locations = np.argsort(
        -abs_ori_dct, axis=None
    )  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [
        (val // rows, val % rows) for val in locations
    ]  # locations as (x,y) coordinates

    # Empy array that will contain the extracted watermark
    w_ex = np.zeros(MARK_SIZE, dtype=np.float64)

    start_index = MID_FREQ_START
    end_index = start_index + MARK_SIZE

    # Extract the watermark
    # for idx, loc in enumerate(locations[1 : MARK_SIZE + 1]):
    for idx, loc in enumerate(locations[start_index:end_index]):
        if params.strategy == EmbeddingStrategy.ADDITIVE:
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / params.alpha
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (params * ori_dct[loc])

    return w_ex


def similarity(watermark_1, watermarked_2):
    s = np.sum(np.multiply(watermark_1, watermarked_2)) / (
        np.sqrt(np.sum(np.multiply(watermark_1, watermark_1)))
        * np.sqrt(np.sum(np.multiply(watermarked_2, watermarked_2)))
    )
    return s
