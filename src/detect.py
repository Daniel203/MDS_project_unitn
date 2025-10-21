from embed import EmbedParameters, EmbeddingStrategy
from cv2.typing import MatLike
import numpy as np
from scipy.fft import dct


def extract_watermark(image: MatLike, watermarked: MatLike, params: EmbedParameters):
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Generate a watermark
    mark_size = 1024
    w_ex = np.zeros(mark_size, dtype=np.float64)  # TODO: hardcoded watermark size, but it's always 1024 so I don't know if we really need to make it a parameter

    # Embed the watermark
    for idx, loc in enumerate(locations[1:mark_size+1]):
        if params.strategy == EmbeddingStrategy.ADDITIVE:
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / params.alpha
        elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / (params*ori_dct[loc])

    return w_ex
