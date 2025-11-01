from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time

import cv2
import numpy as np
import pandas as pd

import detection_ACME as detection
from attacks import _awgn, _blur, _jpeg, _median, _resize, _sharpen


def attack_worker(
    worker_id: int,
    original_image_path: str,
    original_image_name: str,
    attacked_group: str,
    output_path: str,
    watermarked_image: str,
    result_queue: mp.Queue,
):
    """
    This worker will run until its parent process is terminated.
    It finds candidate solutions and puts them in the queue.
    """
    print(f"[Worker {worker_id}] Started.")
    np.random.seed(os.getpid() + int(time.time()))

    while True:
        image_to_attack = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
        if image_to_attack is None:
            print(f"[Worker {worker_id}] Error: Could not read image.")
            time.sleep(1)
            continue

        wpsnr = 100
        detected = 1
        attack_history = []

        while wpsnr > 35 and detected == 1:
            attack_types = ["awgn", "blur", "sharp", "jpeg", "resize", "median"]
            attack = str(np.random.choice(attack_types))
            param_str = ""

            if attack == "awgn":
                param = float(np.random.uniform(5.0, 20.0))
                param_str = f"AWGN (std={param:.2f})"
                attacked_image = _awgn(image_to_attack, param)
            elif attack == "blur":
                param = int(np.random.choice([3, 5, 7]))
                param_str = f"BLUR (ksize={param})"
                attacked_image = _blur(image_to_attack, param)
            elif attack == "sharp":
                param_str = "SHARP"
                attacked_image = _sharpen(image_to_attack)
            elif attack == "jpeg":
                param = int(np.random.randint(30, 91))
                param_str = f"JPEG (quality={param})"
                attacked_image = _jpeg(image_to_attack, param)
            elif attack == "resize":
                param = float(np.random.uniform(0.5, 1.5))
                param_str = f"RESIZE (scale={param:.2f})"
                attacked_image = _resize(image_to_attack, param)
            elif attack == "median":
                param = int(np.random.choice([3, 5, 7]))
                param_str = f"MEDIAN (ksize={param})"
                attacked_image = _median(image_to_attack, param)
            else:
                continue

            attack_history.append(param_str)

            # This is the temporary file for the detection function
            attacked_image_path = f"{output_path}/worker_ACME_{attacked_group}_{original_image_name}_{worker_id}.bmp"
            cv2.imwrite(attacked_image_path, attacked_image)

            detected, wpsnr = detection.detection(
                original_image_path, watermarked_image, attacked_image_path
            )

            print(
                f"[Worker {worker_id}] Att: {param_str}, WPSNR: {wpsnr:.2f}, Det: {detected}"
            )
            image_to_attack = attacked_image

            if detected == 0 and wpsnr >= 35:
                print(f"🎉 [Worker {worker_id}] Found a candidate! WPSNR: {wpsnr:.2f}")
                solution_filename = f"ACME_{attacked_group}_{original_image_name}_{worker_id}_{time.time_ns()}.bmp"
                solution_path = os.path.join(output_path, solution_filename)

                try:
                    # Save the successful image (which is in memory) to the new unique path
                    cv2.imwrite(solution_path, attacked_image)
                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Error saving solution file {solution_path}: {e}"
                    )
                    continue  # Skip this solution

                # Put (wpsnr, new_unique_path, history) into queue
                result_queue.put((wpsnr, solution_path, attack_history))

                break  # Found a solution, break inner loop to start a new walk

            if wpsnr <= 35:
                break


def run_parallel_search(
    original_image_path: str,
    watermarked_image_path: str,
    original_image_name: str,
    attacked_group: str,
    output_path: str,
    num_workers: int,
    search_duration_seconds: int,
):
    """
    Launches workers, runs them for a duration, and saves
    the top 5 best results.
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists("input"):
        os.makedirs("input")
    if not os.path.exists(original_image_path):
        raise ValueError("Original image not found")
    if not os.path.exists(watermarked_image_path):
        raise ValueError("Watermarked image not found")

    print(f"Starting {num_workers} parallel workers...")
    print(f"Searching for the optimal attack for {search_duration_seconds} seconds.")

    with mp.Manager() as manager:
        result_queue = manager.Queue()
        processes = []

        for i in range(num_workers):
            p = mp.Process(
                target=attack_worker,
                args=(
                    i,
                    original_image_path,
                    original_image_name,
                    attacked_group,
                    output_path,
                    watermarked_image_path,
                    result_queue,
                ),
            )
            processes.append(p)
            p.start()

        start_time = time.time()
        while (time.time() - start_time) < search_duration_seconds:
            time.sleep(1)
            print(
                f"Searching... {int(time.time() - start_time)}s / {search_duration_seconds}s"
            )

        print("\n--- TIME'S UP ---")
        print("Terminating workers and collecting results...")

        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

        print("Cleaning up temporary worker files...")
        for i in range(num_workers):
            temp_file = f"{output_path}/worker_ACME_{attacked_group}_{original_image_name}_{i}.bmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_file}: {e}")

        print("All workers stopped.")

        all_solutions = []
        while not result_queue.empty():
            try:
                all_solutions.append(result_queue.get_nowait())
            except queue.Empty:
                break

        if not all_solutions:
            print("\n--- NO SOLUTIONS FOUND ---")
            print("No solutions were found by any worker in the given time.")
            return

        print(f"\nFound {len(all_solutions)} total solutions. Sorting for top 5...")

        all_solutions.sort(key=lambda x: x[0], reverse=True)

        # Keep only the top 5 solutions
        top_solutions = all_solutions[:5]
        top_paths_to_keep = {
            path for _, path, _ in top_solutions
        }  # Set for fast lookup

        print(f"\n--- TOP {len(top_solutions)} ATTACKS ---")

        for i, (wpsnr, path, history) in enumerate(top_solutions):
            print(f"\n--- Rank {i+1} ---")
            print(f"  WPSNR: {wpsnr:.2f}")
            print(f"  Image: {path}")
            print(f"  Attack Sequence ({len(history)} steps):")
            for step_idx, step in enumerate(history):
                print(f"    Step {step_idx+1}: {step}")

        # Clean up unnecessary images from output path
        print("\nCleaning up non-top solution files...")
        cleaned_count = 0
        for _, path, _ in all_solutions:
            if path not in top_paths_to_keep:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Warning: Could not remove file {path}: {e}")

        print(f"Cleanup complete. Removed {cleaned_count} extra files.")
        print(f"The {len(top_solutions)} best images are saved in '{output_path}/'.")

        return top_solutions


def save_solutions_csv(top_solutions, output_filename):
    try:
        # Convert the list of tuples into a pandas DataFrame
        df = pd.DataFrame(
            top_solutions, columns=["WPSNR", "FilePath", "AttackHistory_List"]
        )

        # Convert the list of attack steps into a single, human-readable string
        # using " | " as a separator.
        df["AttackSequence"] = df["AttackHistory_List"].apply(lambda x: " | ".join(x))

        # Create the final DataFrame to save, dropping the original list column
        df_to_save = df.drop(columns=["AttackHistory_List"])

        # Save the DataFrame to a CSV file
        df_to_save.to_csv(output_filename, index=False)

        print(f"Successfully saved results to {output_filename}.")
        print("\nCSV content preview:")
        print(df_to_save.head())

    except Exception as e:
        print(f"Error: Could not save CSV file. {e}")


if __name__ == "__main__":
    WORKERS = os.cpu_count()
    if WORKERS is None:
        WORKERS = 4

    SEARCH_DURATION_SECONDS = 30

    ORIGINAL_IMAGE_PATH = "input/0036.bmp"
    WATERMARKED_IMAGE_PATH = "output/watermarked_image.bmp"
    ORIGINAL_FILE_NAME = "0036"
    ATTACKED_GROUP = "TEST"
    OUTPUT_DIR = "output_0036"

    solutions = run_parallel_search(
        ORIGINAL_IMAGE_PATH,
        WATERMARKED_IMAGE_PATH,
        ORIGINAL_FILE_NAME,
        ATTACKED_GROUP,
        OUTPUT_DIR,
        WORKERS,
        SEARCH_DURATION_SECONDS,
    )

    save_solutions_csv(solutions, f"{OUTPUT_DIR}/results.csv")
