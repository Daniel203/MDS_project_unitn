from __future__ import annotations

import cv2
import numpy as np
import detection_ACME as detection

from attacks import _awgn, _blur, _jpeg, _median, _resize, _sharpen


def randomized_attack(original_image: str, watermarked_image: str):
    """
    Pass either the img array or the img path.
    The first evaluated is the image array, so you can't pass both
    """
    stop = False 

    while not stop:
        image_to_attack = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
        if image_to_attack is None:
            raise FileNotFoundError(f"Could not read input image: {watermarked_image}")

        wpsnr = 100
        detected = 1
        attacked_image = None

        while wpsnr > 35 and detected == 1:
            print()

            # Choose an attack type uniformly at random
            attack_types = ["awgn", "blur", "sharp", "jpeg", "resize", "median"]
            attack = str(np.random.choice(attack_types))

            # Select a random parameter appropriate for the chosen attack
            if attack == "awgn":
                # Standard deviation between 5 and 20
                param = float(np.random.uniform(5.0, 20.0))
                print(f"AWGN - {param=}")
                attacked_image = _awgn(image_to_attack, param)
            elif attack == "blur":
                # Odd kernel size from the set {3, 5, 7}
                param = int(np.random.choice([3, 5, 7]))
                print(f"BLUR - {param=}")
                attacked_image = _blur(image_to_attack, param)
            elif attack == "sharp":
                # Unsharp masking requires no parameter
                print(f"SHARP")
                attacked_image = _sharpen(image_to_attack)
            elif attack == "jpeg":
                # JPEG quality between 30 and 90 (inclusive)
                param = int(np.random.randint(30, 91))
                print(f"JPEG - {param=}")
                attacked_image = _jpeg(image_to_attack, param)
            elif attack == "resize":
                # Resize scale between 0.5 and 1.5
                param = float(np.random.uniform(0.5, 1.5))
                print(f"RESIZE - {param=}")
                attacked_image = _resize(image_to_attack, param)
            elif attack == "median":
                # Median filter kernel size from the set {3, 5, 7}
                param = int(np.random.choice([3, 5, 7]))
                print(f"MEDIAN - {param=}")
                attacked_image = _median(image_to_attack, param)
            else:
                # This branch should never be reached because the attack type
                # is drawn from a fixed list.  It is included for completeness.
                raise ValueError(f"Unknown attack type: {attack}")

            attacked_image_path = "output/attacked_image.bmp"
            cv2.imwrite(attacked_image_path, attacked_image)

            # Detect
            detected, wpsnr = detection.detection(original_image, watermarked_image, attacked_image_path)

            print(f"{wpsnr=}")
            print(f"{detected=}")
            image_to_attack = attacked_image

            if detected == 0 and wpsnr >= 35:
                stop = True

if __name__ == "__main__":
    randomized_attack("input/0036.bmp", "output/0036_watermarked.bmp")
