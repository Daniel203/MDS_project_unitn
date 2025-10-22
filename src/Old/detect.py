# from embed import EmbedParameters, EmbeddingStrategy
# from cv2.typing import MatLike
# import numpy as np
# from scipy.fft import dct


# def extract_watermark(image: MatLike, watermarked: MatLike, params: EmbedParameters):
#     ori_dct = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
#     wat_dct = dct(dct(watermarked, axis=0, norm="ortho"), axis=1, norm="ortho")

#     # Get the locations of the most perceptually significant components
#     ori_dct = abs(ori_dct)
#     wat_dct = abs(wat_dct)
#     locations = np.argsort(
#         -ori_dct, axis=None
#     )  # - sign is used to get descending order
#     rows = image.shape[0]
#     locations = [
#         (val // rows, val % rows) for val in locations
#     ]  # locations as (x,y) coordinates

#     # Generate a watermark
#     mark_size = 1024
#     w_ex = np.zeros(mark_size, dtype=np.float64)
#     # TODO: hardcoded watermark size, but it's always 1024 so I don't know if we really need to make it a parameter

#     # Embed the watermark
#     for idx, loc in enumerate(locations[1 : mark_size + 1]):
#         if params.strategy == EmbeddingStrategy.ADDITIVE:
#             w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / params.alpha
#         elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
#             w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (params * ori_dct[loc])

#     return w_ex


# def similarity(X, X_star):
#     s = np.sum(np.multiply(X, X_star)) / (
#         np.sqrt(np.sum(np.multiply(X, X)))
#         * np.sqrt(np.sum(np.multiply(X_star, X_star)))
#     )
#     return s
from embed import EmbedParameters, EmbeddingStrategy
from cv2.typing import MatLike
import numpy as np
from scipy.fft import dct

def dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")

def build_midband_mask(h: int, w: int, low_cutoff: float, high_cutoff: float) -> np.ndarray:
    # normalized “distance” from DC (0,0)
    U, V = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    D = np.sqrt(U**2 + V**2) / np.sqrt(h**2 + w**2)
    mask = (D > low_cutoff) & (D < high_cutoff)
    mask[0,0] = False  # never use DC
    return mask

def select_locations_by_magnitude(ori_dct: np.ndarray, mask: np.ndarray, seed: int | None, k: int):
    h, w = ori_dct.shape
    mag = np.abs(ori_dct) * mask
    flat = np.argsort(-mag, axis=None)  # sort by magnitude (desc)
    # map flat -> (r,c) using cols, not rows
    locs = [(i // w, i % w) for i in flat if mag.flat[i] > 0]
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(locs)  # must match embed’s permutation step
    return locs[:k]

def extract_watermark(image: MatLike, watermarked: MatLike, params: EmbedParameters):
    # 1) DCTs (signed values preserved for subtraction)
    ori_d = dct2(image.astype(np.float64))
    wat_d = dct2(watermarked.astype(np.float64))

    # 2) Medium-frequency mask (must match embed)
    h, w = ori_d.shape
    # Reuse the exact cutoffs/shape used at embed time:
    low  = getattr(params, "low_cutoff", 0.10)     # e.g., 10%
    high = getattr(params, "high_cutoff", 0.60)    # e.g., 60%
    mask = build_midband_mask(h, w, low, high)

    # 3) Location selection identical to embed
    mark_size = getattr(params, "mark_size", 1024)
    seed = getattr(params, "seed", None)  # same key used in embedding (if any)
    locations = select_locations_by_magnitude(ori_d, mask, seed, mark_size)

    # 4) Reconstruct watermark using the same model as embed
    w_ex = np.zeros(mark_size, dtype=np.float64)
    eps = 1e-12

    if params.strategy == EmbeddingStrategy.ADDITIVE:
        # embed: wat = ori + alpha * w  =>  w = (wat - ori)/alpha
        for i, (r, c) in enumerate(locations):
            w_ex[i] = (wat_d[r, c] - ori_d[r, c]) / params.alpha

    elif params.strategy == EmbeddingStrategy.MULTIPLICATIVE:
        # embed: wat = ori * (1 + alpha * w)  =>  w = (wat - ori)/(alpha * ori)
        for i, (r, c) in enumerate(locations):
            w_ex[i] = (wat_d[r, c] - ori_d[r, c]) / (params.alpha * (ori_d[r, c] + eps))

    else:
        raise ValueError(f"Unsupported strategy: {params.strategy}")

    return w_ex

def similarity(X, X_star):
    # cosine similarity
    num = np.sum(X * X_star)
    den = np.sqrt(np.sum(X * X)) * np.sqrt(np.sum(X_star * X_star))
    return num / (den + 1e-12)
