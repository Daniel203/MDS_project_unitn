import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from attack import randomized_attack, attacks
from constraints import (ALPHA, INPUT_DIR, MARK_SIZE, OUTPUT_DIR, THRESH,
                         WATERMARK_NAME)
from detect import extract_watermark, similarity
import detect
from embed import EmbeddingStrategy, EmbedParameters, embed_watermark
from roc import generate_roc_curve_plot
from wpsnr import wpsnr


def embedding(input1: str, input2: str) -> MatLike | None:
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    watermark = np.load(input2)
    params = EmbedParameters(ALPHA, EmbeddingStrategy.ADDITIVE)
    output1 = embed_watermark(image, watermark, params)
    return output1


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

    params = EmbedParameters(ALPHA, EmbeddingStrategy.ADDITIVE)

    watermark_original = extract_watermark(image_original, image_watermarked, params)
    watermark_attacked = extract_watermark(image_original, image_attacked, params)

    output1 = 1 if similarity(watermark_original, watermark_attacked) > THRESH else 0
    output2 = wpsnr(image_watermarked, image_attacked)

    # return output1, output2
    return output1, output2


def full_workflow():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_filenames = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp")]

    for filename in image_filenames:
        image_path = os.path.join(INPUT_DIR, filename)
        watermarked_image = embedding(image_path, WATERMARK_NAME)
        if watermarked_image is None:
            print("Cannot find the watermarked image!")
            continue

        # Save with modified name
        name, _ = os.path.splitext(filename)
        output_path_watermarked = os.path.join(OUTPUT_DIR, f"{name}_watermarked.bmp")
        cv2.imwrite(output_path_watermarked, watermarked_image)
        print(f"Watermarked image saved: {output_path_watermarked}")

        # Calculate and print WPSNR
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        wpsnr_value = wpsnr(image, watermarked_image)
        print(f"WPSNR embedded image: {wpsnr_value:.4f}")

        attacked_image = randomized_attack(watermarked_image)
        if attacked_image is None:
            continue

        # save the image as
        output_path_attacked = os.path.join(OUTPUT_DIR, f"{name}_attacked.bmp")
        cv2.imwrite(output_path_attacked, attacked_image)
        print(f"Attacked image saved: {output_path_attacked}")

        detect1, detect2 = detection(
            image_path, output_path_watermarked, output_path_attacked
        )

        print(f"watermark detected: {'yes' if detect1 == 1 else 'no'}")
        print(f"WPSNR attacked image: {detect2}")
        print()


def roc_curve():
    images = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".bmp"):
            image_path = os.path.join(INPUT_DIR, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)

    generate_roc_curve_plot(images, np.load(WATERMARK_NAME))


def test_false_positives(n_images_to_test=5, n_fakes_per_image=1000):
    """
    Tests the system's False Positive Rate (FPR).
    This simulates an attacker guessing random watermarks and
    checks how often we get a false detection.
    """
    print("--- Starting False Positive Test ---")

    # Get a few images
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp")]
    images_to_test = image_files[:n_images_to_test]

    # Load the *real* watermark to create the attacked images
    real_watermark = np.load(WATERMARK_NAME)

    total_tests = 0
    total_false_positives = 0
    params = EmbedParameters(ALPHA, EmbeddingStrategy.ADDITIVE)

    # --- START OF MODIFICATION ---
    all_fake_scores = []  # List to store all fake scores for plotting
    # --- END OF MODIFICATION ---

    for filename in images_to_test:
        image_path = os.path.join(INPUT_DIR, filename)
        image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_original is None:
            continue

        print(f"Testing on image: {filename}")

        # 1. Create a "real" attacked image
        image_watermarked = embed_watermark(image_original, real_watermark, params)
        image_attacked = randomized_attack(image_watermarked)  # Attack it

        # 2. Extract the (noisy) watermark
        watermark_extracted = extract_watermark(image_original, image_attacked, params)

        # 3. Test it against many FAKE watermarks
        for _ in range(n_fakes_per_image):
            # Create a random, bipolar fake watermark
            # We use MARK_SIZE which is 1024
            watermark_fake = np.random.choice([-1.0, 1.0], MARK_SIZE)

            # Check similarity
            score = similarity(watermark_fake, watermark_extracted)

            # --- START OF MODIFICATION ---
            all_fake_scores.append(score)  # Store the score
            # --- END OF MODIFICATION ---

            # If the random guess scores higher than our threshold, it's a False Positive
            if score > THRESH:
                total_false_positives += 1

            total_tests += 1

    # Calculate the final False Positive Rate (FPR)
    if total_tests > 0:
        fpr = total_false_positives / total_tests
        print("\n--- False Positive Test Results ---")
        print(
            f"Total Tests Run: {total_tests} (Images: {len(images_to_test)}, Fakes: {n_fakes_per_image})"
        )
        print(f"Total False Positives Found: {total_false_positives}")
        print(f"False Positive Rate (FPR): {fpr * 100:.4f}%")
        print(f"(Your detection threshold 'THRESH' is {THRESH})")

        # --- START OF PLOTTING CODE ---
        print("\nGenerating similarity score plot...")

        plt.figure(figsize=(10, 5))
        # Plot all the individual scores (will look like noise)
        plt.plot(all_fake_scores, alpha=0.5, label="Fake Similarity Scores")

        # Plot the red threshold line
        plt.hlines(
            THRESH,
            0,
            total_tests,
            "r",
            linestyle="dashed",
            label=f"Threshold ({THRESH})",
        )

        plt.title("Similarity Scores of Fake Watermarks vs. Threshold")
        plt.xlabel("Test Sample Index")
        plt.ylabel("Similarity Score")
        plt.legend()
        plt.grid(True)

        plot_filename = "fpr_plot.png"
        plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Score plot saved to: {plot_filename}")

    else:
        print("No images found to test.")


def test():
    image_watermarked = embedding("images/0000.bmp", WATERMARK_NAME)
    cv2.imwrite("output/test_watermarked.bmp", image_watermarked)

    cv2.imwrite("output/test_attacked.bmp", image_watermarked)
    # attacked_image = attacks("output/test_attacked.bmp", "median", [5])
    # cv2.imwrite("output/test_attacked.bmp", attacked_image)
    attacked_image = attacks("output/test_attacked.bmp", "blur", [5])
    cv2.imwrite("output/test_attacked.bmp", attacked_image)
    # attacked_image = attacks("output/test_attacked.bmp", "jpeg", [10])
    # cv2.imwrite("output/test_attacked.bmp", attacked_image)
    # attacked_image = attacks("output/test_attacked.bmp", "sharpen", [10])
    # cv2.imwrite("output/test_attacked.bmp", attacked_image)
    # attacked_image = attacks("output/test_attacked.bmp", "resize", [80])
    # cv2.imwrite("output/test_attacked.bmp", attacked_image)
    # attacked_image = attacks("output/test_attacked.bmp", "median", [81])
    # cv2.imwrite("output/test_attacked.bmp", attacked_image)

    cv2.imwrite("output/test_attacked.bmp", attacked_image)
    detect1, detect2 = detection(
        "images/0000.bmp", "output/test_watermarked.bmp", "output/test_attacked.bmp"
    )

    print(f"watermark detected: {'yes' if detect1 == 1 else 'no'}")
    print(f"WPSNR attacked image: {detect2}")

if __name__ == "__main__":
    # test()
    roc_curve()
    # test_false_positives(n_images_to_test=100)
