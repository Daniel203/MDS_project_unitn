from cv2.typing import MatLike
import numpy as np 
import cv2
from embed import embed_watermark, EmbedParameters, EmbeddingStrategy
from detect import extract_watermark
import os
from wpsnr import wpsnr

def embedding(input1: str, input2: str) -> MatLike|None:
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None 

    watermark = np.load(input2)
    params = EmbedParameters(0.05, EmbeddingStrategy.ADDITIVE)
    output1 = embed_watermark(image, watermark, params)
    return output1


def detection(input1: str, input2: str, input3: str) -> tuple[int, float]:
    img_or = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    img_wm = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    img_at = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    params = EmbedParameters(0.05, EmbeddingStrategy.ADDITIVE)

    wm_or = extract_watermark(img_wm, img_or, params)
    wm_ex = extract_watermark(img_at, img_or, params)

    # TODO: impl similarity
    # output1 = similarity(wm_or, wm_ex) > THRESH
    # output1 = 1 if output1 else 0
    # output2 = wPSNR(img_wm, img_at)

    # return output1, output2
    return 0, 0.0 


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
                print("Cannot find the watermarked image!");
                continue

            # Save with modified name
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_watermarked.bmp")
            cv2.imwrite(output_path, watermarked_image)
            print(f"Watermarked image saved: {output_path}")

            # Calculate and print WPSNR
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 

            wpsnr_value = wpsnr(image, watermarked_image)
            print(f"WPSNR after: {wpsnr_value:.4f}\n")
    

if __name__ == "__main__":
    main()
