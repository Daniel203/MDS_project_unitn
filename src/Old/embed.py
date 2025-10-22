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

    # # Get the locations of the most perceptually significant components
    # sign = np.sign(ori_dct)
    # ori_dct = abs(ori_dct)
    # locations = np.argsort(
    #     -ori_dct, axis=None
    # )  # - sign is used to get descending order
    # rows = image.shape[0]
    # locations = [
    #     (val // rows, val % rows) for val in locations
    # ]  # locations as (x,y) coordinates

    # Get the DCT and its absolute value
    sign = np.sign(ori_dct)
    ori_dct = np.abs(ori_dct)

    rows, cols = ori_dct.shape

    # Define a medium frequency mask
    # Low frequencies are near (0,0), high near (rows-1, cols-1)
    low_cutoff = 0.1  # 10% from top-left (low freq)
    high_cutoff = 0.6  # 60% from top-left (exclude the highest freq)

    # Compute frequency distance grid
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2) / np.sqrt(rows**2 + cols**2)  # normalized distance [0,1]

    # Create mask for medium frequencies only
    medium_mask = (D > low_cutoff) & (D < high_cutoff)

    # Apply mask: zero out coefficients outside medium frequencies
    masked_dct = ori_dct * medium_mask

    # Sort remaining medium-frequency coefficients (descending)
    locations = np.argsort(-masked_dct, axis=None)
    locations = [(val // cols, val % cols) for val in locations if masked_dct.flat[val] > 0]

    # Restore original signs if needed
    # ori_dct = masked_dct * sign


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
