"""Image processing attacks for watermark robustness testing.

The functions in this module implement the set of permitted attacks
described in the competition rules: additive white Gaussian noise
(AWGN), Gaussian blurring, sharpening (unsharp masking), JPEG
compression, resizing and median filtering.  All functions operate on
grayscale images and return a NumPy array without writing any files
to disk.  The main entry point is :func:`attacks` which applies one
or more attacks in sequence.

Use this script from your demo code or your own experiments.  When
submitting to the teaching assistants you should include a brief
README explaining how to use the functions; see the ``README.md``
generated alongside this file.
"""

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Union
import numpy as np
import cv2


def _awgn(img: np.ndarray, std: float) -> np.ndarray:
    """Additive white Gaussian noise with given standard deviation."""
    noise = np.random.normal(0.0, std, img.shape)
    attacked = img.astype(np.float64) + noise
    return np.clip(attacked, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur with a square kernel of size ``ksize``.  ``ksize`` must
    be an odd positive integer."""
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("Blur kernel size must be a positive odd integer")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _sharpen(img: np.ndarray) -> np.ndarray:
    """Sharpen the image using an unsharp masking filter."""
    # Simple 3×3 sharpening kernel.  This increases high‑frequency
    # components and attenuates low‑frequency components.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress and decompress the image using JPEG with the given quality.

    Quality should lie between 1 (very low) and 100 (very high).  The
    compression step is performed in memory using OpenCV's imencode
    function.
    """
    quality = int(max(1, min(quality, 100)))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # Encode to JPEG in memory
    success, buffer = cv2.imencode('.jpg', img, encode_param)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    # Decode back into a NumPy array
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return decoded


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize the image by the given scaling factor and then restore
    it back to the original size.  The intermediate resizing may
    destroy watermark information at subpixel accuracy."""
    if scale <= 0:
        raise ValueError("Scale must be positive")
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)


def _median(img: np.ndarray, ksize: int) -> np.ndarray:
    """Apply a median filter with the given kernel size.  ``ksize`` must
    be a positive odd integer."""
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("Median kernel size must be a positive odd integer")
    return cv2.medianBlur(img, ksize)


def attacks(input1: str,
            attack_name: Union[str, Sequence[str]],
            param_array: Union[Sequence[Union[float, int]], Sequence[Sequence[Union[float, int]]]]) -> np.ndarray:
    """Apply one or more image processing attacks to a watermarked image.

    The attack names can be one of: ``'awgn'``, ``'blur'``, ``'sharp'`` or
    ``'sharpen'``, ``'jpeg'``, ``'resize'`` and ``'median'``.  When
    multiple attacks are specified the image is processed by each
    attack in order.  The parameters should either be a flat list
    specifying the parameter for each attack or a list of lists.

    Args:
        input1: Path to the watermarked image to attack.
        attack_name: A single attack name or a list/tuple of attack
            names specifying the processing pipeline.
        param_array: A sequence of parameters corresponding to each
            attack.  If a single attack is specified this may be a
            single value.  For multiple attacks it must be a sequence
            with length equal to the number of attacks.

    Returns:
        The attacked image as a NumPy array.
    """
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input1}")

    # Ensure we have a list of attacks and parameters
    if isinstance(attack_name, (str, bytes)):
        attacks_list: List[str] = [str(attack_name).lower()]
    else:
        attacks_list = [str(a).lower() for a in attack_name]

    # Normalise the parameter array into a list of lists.  If a flat
    # list is passed for multiple attacks we assume each element
    # corresponds to one attack.  If nested lists/tuples are passed they
    # are used directly.
    if not isinstance(param_array, (list, tuple)):
        param_list: List[Tuple] = [(param_array,)] * len(attacks_list)
    else:
        if len(attacks_list) == 1:
            # Single attack can accept a flat list of parameters
            param_list = [(param_array,)] if not isinstance(param_array[0], (list, tuple)) else param_array  # type: ignore
        else:
            # Expect one parameter (or parameter list) per attack
            if len(param_array) != len(attacks_list):
                raise ValueError("Length of param_array must match number of attacks")
            # Wrap each element in a tuple if it isn't already a sequence
            param_list = []
            for p in param_array:
                if isinstance(p, (list, tuple)):
                    param_list.append(tuple(p))
                else:
                    param_list.append((p,))

    # Apply each attack sequentially
    output = img.copy()
    for name, params in zip(attacks_list, param_list):
        if name == 'awgn':
            sigma = float(params[0])
            output = _awgn(output, sigma)
        elif name == 'blur':
            ksize = int(params[0])
            output = _blur(output, ksize)
        elif name in ('sharp', 'sharpen'):
            output = _sharpen(output)
        elif name == 'jpeg':
            quality = int(params[0])
            output = _jpeg(output, quality)
        elif name == 'resize':
            scale = float(params[0])
            output = _resize(output, scale)
        elif name == 'median':
            ksize = int(params[0])
            output = _median(output, ksize)
        else:
            raise ValueError(f"Unknown attack name: {name}")
    return output