from cv2.typing import MatLike
from scipy.fft import dct, idct
import numpy as np
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from typing import cast


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
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(
        -ori_dct, axis=None
    )  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [
        (val // rows, val % rows) for val in locations
    ]  # locations as (x,y) coordinates

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for loc, mark_val in zip(locations[1:], watermark):
        if params.strategy == EmbeddingStrategy.ADDITIVE:
            watermarked_dct[loc] += params.alpha * mark_val
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            watermarked_dct[loc] *= 1 + (params.alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = np.uint8(
        idct(idct(watermarked_dct, axis=1, norm="ortho"), axis=0, norm="ortho")
    )

    watermarked = cast(MatLike, watermarked)
    return watermarked
