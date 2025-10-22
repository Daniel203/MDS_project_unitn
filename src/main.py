from cv2.typing import MatLike
import numpy as np
import cv2
import os
from embedding_code import embed_watermark
from wpsnr import wpsnr
# from embed import embed_watermark, EmbedParameters, EmbeddingStrategy
from embed import EmbedParameters, EmbeddingStrategy

from detect import extract_watermark, similarity
from attack import jpeg_compression

THRESH = 0.55


def embedding(input1: str, input2: str) -> MatLike | None:
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    watermark = np.load(input2)
    params = EmbedParameters(0.05, EmbeddingStrategy.ADDITIVE)
    # output1 = embed_watermark(image, watermark, params)
    wm_img, wm_seq, _  = embed_watermark(input1)
    base, ext = os.path.splitext(input1)
    out_path = f"{base}_wm{ext}"
    cv2.imwrite(out_path, wm_img)

    
    return wm_img


def detection(input1: str, input2: str, input3: str) -> tuple[int, float]:
    image_original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    image_watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    image_attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    if image_original is None:
        raise ValueError(f"Failed to load original image from {input1}")
    if image_watermarked is None:
        raise ValueError(f"Failed to load watermarked image from {input2}")
    if image_attacked is None:
        raise ValueError(f"Failed to load attacked image from {input3}")

    params = EmbedParameters(0.05, EmbeddingStrategy.ADDITIVE)

    watermark_original = extract_watermark(image_watermarked, image_original, params)
    watermark_extracted = extract_watermark(image_attacked, image_original, params)


    output1 = similarity(watermark_original, watermark_extracted) > THRESH
    output1 = 1 if output1 else 0
    output2 = wpsnr(image_watermarked, image_attacked)

    # return output1, output2
    return output1, output2


def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    watermark_name = "watermark.npy"

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".bmp"):
            image_path = os.path.join(input_dir, filename)
            watermarked_image = embedding(image_path, watermark_name)

            if watermarked_image is None:
                print("Cannot find the watermarked image!")
                continue

            # Save with modified name
            name, _ = os.path.splitext(filename)
            output_path_watermarked = os.path.join(output_dir, f"{name}_watermarked.bmp")
            cv2.imwrite(output_path_watermarked, watermarked_image)
            print(f"Watermarked image saved: {output_path_watermarked}")

            # Calculate and print WPSNR
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            wpsnr_value = wpsnr(image, watermarked_image)
            print(f"WPSNR embedded image: {wpsnr_value:.4f}")

            attacked_image = jpeg_compression(watermarked_image, 80)
            if attacked_image is None:
                continue

            # save the image as 
            output_path_attacked = os.path.join(output_dir, f"{name}_attacked.bmp")
            cv2.imwrite(output_path_attacked, attacked_image)
            print(f"Attacked image saved: {output_path_attacked}")

            detect1, detect2 = detection(image_path, output_path_watermarked, output_path_attacked)

            print(f"watermark detected: {'yes' if detect1 == 1 else 'no'}")
            print(f"WPSNR attacked image: {detect2}")
            print()


if __name__ == "__main__":
    main()
