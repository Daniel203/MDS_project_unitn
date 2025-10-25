from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.fft import dct, idct

from constraints import MID_FREQ_START, MARK_SIZE


class EmbeddingStrategy(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class EmbedParameters:
    alpha: float
    strategy: EmbeddingStrategy


def embed_watermark(
    image: MatLike, watermark: NDArray[np.uint8], params: EmbedParameters
) -> MatLike:
    # Get the DCT transform of the image
    ori_dct = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")

    # Get the locations of the most perceptually significant components
    # sign = np.sign(ori_dct)
    abs_dct = abs(ori_dct)
    locations = np.argsort(
        -abs_dct, axis=None
    )  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [
        (val // rows, val % rows) for val in locations
    ]  # locations as (x,y) coordinates

    start_index = MID_FREQ_START
    end_index = start_index + MARK_SIZE

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    # for loc, mark_val in zip(locations[1:], watermark):
    for loc, mark_val in zip(locations[start_index:end_index], watermark):
        # Map values (0, 1) -> (-1, 1)
        mark_val_mapped = (float(mark_val) * 2.0) - 1.0
        if params.strategy == EmbeddingStrategy.ADDITIVE:
            watermarked_dct[loc] += params.alpha * mark_val_mapped
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            watermarked_dct[loc] *= 1 + (params.alpha * mark_val_mapped)

    # Restore sign and o back to spatial domain
    # watermarked_dct *= sign
    watermarked = np.uint8(
        idct(idct(watermarked_dct, axis=1, norm="ortho"), axis=0, norm="ortho")
    )

    watermarked = cast(MatLike, watermarked)
    return watermarked
