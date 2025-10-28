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
    success, buffer = cv2.imencode(".jpg", img, encode_param)
    if not success:
        raise RuntimeError("JPEG encoding failed")

    # Decode back into a NumPy array
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise FileNotFoundError("Decoded image not fonud!")  

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


def randomized_attack(image: np.ndarray | None = None, image_path: str | None = None) -> np.ndarray:
    """
    Pass either the img array or the img path.
    The first evaluated is the image array, so you can't pass both 
    """

    img = None
    if image is not None:
        img = image
    elif image_path is not None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read input image: {image_path}")
    else:
        raise ValueError("Wrong parameters, either image or image_path must be not null!")

    # Choose an attack type uniformly at random
    attack_types = ["awgn", "blur", "sharp", "jpeg", "resize", "median"]
    attack = str(np.random.choice(attack_types))

    # Select a random parameter appropriate for the chosen attack
    if attack == "awgn":
        # Standard deviation between 5 and 20
        param = float(np.random.uniform(5.0, 20.0))
        return _awgn(img, param)
    elif attack == "blur":
        # Odd kernel size from the set {3, 5, 7}
        param = int(np.random.choice([3, 5, 7]))
        return _blur(img, param)
    elif attack == "sharp":
        # Unsharp masking requires no parameter
        return _sharpen(img)
    elif attack == "jpeg":
        # JPEG quality between 30 and 90 (inclusive)
        param = int(np.random.randint(30, 91))
        return _jpeg(img, param)
    elif attack == "resize":
        # Resize scale between 0.5 and 1.5
        param = float(np.random.uniform(0.5, 1.5))
        return _resize(img, param)
    elif attack == "median":
        # Median filter kernel size from the set {3, 5, 7}
        param = int(np.random.choice([3, 5, 7]))
        return _median(img, param)
    else:
        # This branch should never be reached because the attack type
        # is drawn from a fixed list.  It is included for completeness.
        raise ValueError(f"Unknown attack type: {attack}")


def attacks(
    input1: str,
    attack_name: Union[str, Sequence[str]],
    param_array: Union[
        Sequence[Union[float, int]], Sequence[Sequence[Union[float, int]]]
    ],
) -> np.ndarray:
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input1}")

    # Ensure we have a list of attacks and parameters
    if isinstance(attack_name, (str, bytes)):
        attacks_list: List[str] = [str(attack_name).lower()]
    else:
        attacks_list = [str(a).lower() for a in attack_name]

    # Normalise the parameter array into a list.  For a single attack,
    # param_array may be a scalar, a list/tuple of values, or a
    # nested list.  For multiple attacks, param_array must be a
    # sequence with one parameter (or parameter list) per attack.
    if not isinstance(param_array, (list, tuple)):
        # A scalar parameter is applied to all attacks
        param_list: List = [param_array] * len(attacks_list)
    else:
        if len(attacks_list) == 1:
            # Single attack: if the parameter array itself is a list of
            # simple values (e.g. [10]) treat it as the parameter
            # sequence; if it's a nested list (e.g. [[10, 20]]) use it
            # directly.
            if len(param_array) > 0 and not isinstance(param_array[0], (list, tuple)):
                param_list = [param_array]
            else:
                param_list = list(param_array)
        else:
            # Multiple attacks: expect one entry per attack name
            if len(param_array) != len(attacks_list):
                raise ValueError("Length of param_array must match number of attacks")
            param_list = []
            for p in param_array:
                param_list.append(p)

    # Apply each attack sequentially
    output = img.copy()
    for name, param in zip(attacks_list, param_list):
        # Ensure the parameter is indexable
        if isinstance(param, (list, tuple)):
            p = param[0] if len(param) > 0 else None
        else:
            p = param

        if p is None:
            raise ValueError("p cannot be None!") 

        if name == "awgn":
            sigma = float(p)
            output = _awgn(output, sigma)
        elif name == "blur":
            ksize = int(p)
            output = _blur(output, ksize)
        elif name in ("sharp", "sharpen"):
            output = _sharpen(output)
        elif name == "jpeg":
            quality = int(p)
            output = _jpeg(output, quality)
        elif name == "resize":
            scale = float(p)
            output = _resize(output, scale)
        elif name == "median":
            ksize = int(p)
            output = _median(output, ksize)
        else:
            raise ValueError(f"Unknown attack name: {name}")
    return output
