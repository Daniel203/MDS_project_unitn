from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
import pywt
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.fft import dct, idct

from constraints import MARK_SIZE, MID_FREQ_START


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
    # Apply the dwt transform
    coeffs = pywt.dwt2(image, "haar")  # 'haar' is a simple, fast wavelet
    LL, (LH, HL, HH) = coeffs

    # Get the dct transform of LL sub-band
    ori_dct = dct(dct(LL, axis=0, norm="ortho"), axis=1, norm="ortho")

    # Find locations
    abs_dct = abs(ori_dct)
    locations = np.argsort(-abs_dct, axis=None)
    rows = LL.shape[0]  # Use LL sub-band shape
    locations = [(val // rows, val % rows) for val in locations]

    start_index = MID_FREQ_START
    end_index = start_index + MARK_SIZE

    # Embed the watermark onto the absolute values
    watermarked_dct = ori_dct.copy()

    for i, loc in enumerate(locations[start_index:end_index]):
        # Convert the watermark from [0, 1] -> [-1, +1]
        mark_val = (float(watermark[i]) * 2.0) - 1.0

        if params.strategy == EmbeddingStrategy.ADDITIVE:
            watermarked_dct[loc] += params.alpha * mark_val
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            watermarked_dct[loc] *= 1 + (params.alpha * mark_val)

    # Apply the inverse dct to get the modified LL sub-band
    watermarked_LL = idct(
        idct(watermarked_dct, axis=1, norm="ortho"), axis=0, norm="ortho"
    )

    # Reconstruct the full image
    watermarked_coeffs = (watermarked_LL, (LH, HL, HH))
    watermarked_image = pywt.idwt2(watermarked_coeffs, "haar")

    # Clip values to valid 0-255 range
    watermarked = np.clip(watermarked_image, 0, 255)
    watermarked = np.uint8(watermarked)

    watermarked = cast(MatLike, watermarked)
    return watermarked
