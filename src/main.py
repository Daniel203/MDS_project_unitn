import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from attack import attacks, randomized_attack
from constraints import (ALPHA, INPUT_DIR, MARK_SIZE, OUTPUT_DIR, THRESH,
                         WATERMARK_NAME)
from detect import extract_watermark, similarity
from embed import EmbeddingStrategy, EmbedParameters, embed_watermark
from roc import generate_roc_curve_plot
from wpsnr import wpsnr
from concurrent.futures import ProcessPoolExecutor, as_completed


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

def process_image(filename):
    image_path = os.path.join(INPUT_DIR, filename)
    
    image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_original is None:
        return (filename, "failed", "load", 0.0, 0)

    watermarked_image = embedding(image_path, WATERMARK_NAME)
    if watermarked_image is None:
        return (filename, "failed", "embed", 0.0, 0)

    name, _ = os.path.splitext(filename)
    output_path_watermarked = os.path.join(OUTPUT_DIR, f"{name}_watermarked.bmp")
    cv2.imwrite(output_path_watermarked, watermarked_image)
    wpsnr_value = wpsnr(image_original, watermarked_image)

    # --- Attack phase ---
    attack_attempts = 0
    max_attack_attempts = 10
    attacked_image = None
    attack_wpsnr = 0.0

    while attack_attempts < max_attack_attempts:
        current_attacked_image = randomized_attack(watermarked_image)
        if current_attacked_image is None:
            attack_attempts += 1
            continue

        current_wpsnr = wpsnr(watermarked_image, current_attacked_image)
        if current_wpsnr >= 35.0:
            attacked_image = current_attacked_image
            attack_wpsnr = current_wpsnr
            break
        attack_attempts += 1

    if attacked_image is None:
        return (filename, "failed", "attack", 0.0, 0)

    output_path_attacked = os.path.join(OUTPUT_DIR, f"{name}_attacked.bmp")
    cv2.imwrite(output_path_attacked, attacked_image)

    detect1, detect2 = detection(image_path, output_path_watermarked, output_path_attacked)
    detected = 1 if detect1 == 1 else 0

    return (filename, "ok", "done", attack_wpsnr, detected)

def full_workflow_parallel(max_workers=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_filenames = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp")]

    total_images_processed = 0
    total_detected = 0
    total_not_detected = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, f): f for f in image_filenames}

        for future in as_completed(futures):
            filename = futures[future]
            try:
                fname, status, stage, attack_wpsnr, detected = future.result()
                if status != "ok":
                    print(f"{filename}: Failed at stage {stage}")
                    continue

                print(f"{filename}: Done, WPSNR={attack_wpsnr:.2f}, Detected={'yes' if detected else 'no'}")
                total_images_processed += 1
                if detected:
                    total_detected += 1
                else:
                    total_not_detected += 1

            except Exception as e:
                print(f"{filename}: Error {e}")

    print("\n--- Workflow Summary ---")
    print(f"Total images processed: {total_images_processed}")
    print(f"Watermarks detected ('yes'): {total_detected}")
    print(f"Watermarks not detected ('no'): {total_not_detected}")
    if total_images_processed > 0:
        detection_rate = (total_detected / total_images_processed) * 100
        print(f"Detection Rate: {detection_rate:.2f}%")
    print("------------------------\n")



def full_workflow():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_filenames = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp")]

    total_images_processed = 0
    total_detected = 0
    total_not_detected = 0

    for filename in image_filenames:
        image_path = os.path.join(INPUT_DIR, filename)
        
        # --- Load original image ONCE ---
        image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_original is None:
            print(f"Skipping {filename}: Failed to load original image.")
            continue
            
        watermarked_image = embedding(image_path, WATERMARK_NAME)
        if watermarked_image is None:
            print(f"Skipping {filename}: Failed to embed watermark.")
            continue

        # Save with modified name
        name, _ = os.path.splitext(filename)
        output_path_watermarked = os.path.join(OUTPUT_DIR, f"{name}_watermarked.bmp")
        cv2.imwrite(output_path_watermarked, watermarked_image)
        print(f"Watermarked image saved: {output_path_watermarked}")

        wpsnr_value = wpsnr(image_original, watermarked_image)
        print(f"WPSNR embedded image: {wpsnr_value:.4f}")

        attack_attempts = 0
        max_attack_attempts = 10 # Prevent infinite loop
        attacked_image = None
        attack_wpsnr = 0.0

        while attack_attempts < max_attack_attempts:
            # Apply a random attack to the clean watermarked image
            current_attacked_image = randomized_attack(watermarked_image)
            if current_attacked_image is None:
                print("  Attack failed, trying again...")
                attack_attempts += 1
                continue

            # Check the WPSNR of this attack
            current_wpsnr = wpsnr(watermarked_image, current_attacked_image)
            print(f"  Attempt {attack_attempts + 1}: Attack WPSNR = {current_wpsnr:.4f}")

            # Check if the attack is valid (WPSNR >= 35)
            if current_wpsnr >= 35.0:
                attacked_image = current_attacked_image
                attack_wpsnr = current_wpsnr
                print(f"  Valid attack found (WPSNR >= 35).")
                break # Exit the while loop
            else:
                print(f"  Attack invalid (WPSNR < 35), re-attacking...")
                attack_attempts += 1
                
        # Check if we failed to find a valid attack
        if attacked_image is None:
            print(f"Skipping {filename}: Failed to generate a valid attack after {max_attack_attempts} attempts.")
            continue # Skip to the next image file

        # Save the valid attacked image
        output_path_attacked = os.path.join(OUTPUT_DIR, f"{name}_attacked.bmp")
        cv2.imwrite(output_path_attacked, attacked_image)
        print(f"Attacked image saved: {output_path_attacked}")

        # Perform detection using the valid attacked image
        detect1, detect2 = detection(
            image_path, output_path_watermarked, output_path_attacked
        )
        
        # Ensure detect2 matches the calculated attack_wpsnr
        # print(f"WPSNR attacked image (from detection): {detect2}") # Debug line
        print(f"WPSNR attacked image (final): {attack_wpsnr:.4f}") # Print the valid WPSNR

        if detect1 == 1:
            print(f"Watermark detected: yes")
            total_detected += 1
        else:
            print(f"Watermark detected: no")
            total_not_detected += 1
        total_images_processed += 1
        
        print() # Add a blank line for readability

    print("\n--- Workflow Summary ---")
    print(f"Total images processed: {total_images_processed}")
    print(f"Watermarks detected ('yes'): {total_detected}")
    print(f"Watermarks not detected ('no'): {total_not_detected}")
    if total_images_processed > 0:
        detection_rate = (total_detected / total_images_processed) * 100
        print(f"Detection Rate: {detection_rate:.2f}%")
    print("------------------------\n")


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

            all_fake_scores.append(score)  # Store the score

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


def test_attacks_manually():
    original_image_path = "images/0031.bmp"
    watermarked_image_path = "output/test_watermarked.bmp"
    attacked_image_path = "output/test_attacked.bmp"
    image_watermarked = embedding(original_image_path, WATERMARK_NAME)
    if image_watermarked is None:
        return

    cv2.imwrite(watermarked_image_path, image_watermarked)

    cv2.imwrite(attacked_image_path, image_watermarked)
    # attacked_image = attacks(attacked_image_path, "median", [5])
    # cv2.imwrite(attacked_image_path, attacked_image)
    # attacked_image = attacks(attacked_image_path, "blur", [5])
    # cv2.imwrite(attacked_image_path, attacked_image)
    # attacked_image = attacks(attacked_image_path, "jpeg", [10])
    # cv2.imwrite(attacked_image_path, attacked_image)
    # attacked_image = attacks(attacked_image_path, "sharpen", [10])
    # cv2.imwrite(attacked_image_path, attacked_image)
    # attacked_image = attacks(attacked_image_path, "resize", [80])
    # cv2.imwrite(attacked_image_path, attacked_image)
    attacked_image = attacks(attacked_image_path, "awgn", [30])
    cv2.imwrite(attacked_image_path, attacked_image)
    attacked_image = attacks(attacked_image_path, "median", [5])
    cv2.imwrite(attacked_image_path, attacked_image)
    # attacked_image = attacks(attacked_image_path, "sharpen", [5])
    # cv2.imwrite(attacked_image_path, attacked_image)

    cv2.imwrite(attacked_image_path, attacked_image)
    detect1, detect2 = detection(
        original_image_path, watermarked_image_path, attacked_image_path
    )

    print(f"watermark detected: {'yes' if detect1 == 1 else 'no'}")
    print(f"WPSNR attacked image: {detect2}")


if __name__ == "__main__":
    full_workflow_parallel(max_workers=10)
    # test_attacks_manually()
    # roc_curve()
    # test_false_positives(n_images_to_test=100)
