from __future__ import annotations

from __future__ import annotations
import numpy as np
import cv2
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import detection_ACME as detection

from attacks import _awgn, _blur, _jpeg, _median, _resize, _sharpen, randomized_attack

def attack_worker(
    worker_id: int,
    original_image: str,
    watermarked_image: str,
    result_queue: mp.Queue,
):
    """
    This worker will run until its parent process is terminated.
    It no longer stops when a solution is found.
    """
    print(f"[Worker {worker_id}] Started.")
    np.random.seed(os.getpid() + int(time.time()))
    
    # The worker will run in this loop indefinitely
    while True:
        image_to_attack = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
        if image_to_attack is None:
            print(f"[Worker {worker_id}] Error: Could not read image.")
            time.sleep(1) # Avoid spamming errors
            continue

        wpsnr = 100
        detected = 1
        attack_history = []
        
        while wpsnr > 35 and detected == 1:
            # This inner loop is still sequential, as it should be
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
            attacked_image_path = f"output/attacked_image_w{worker_id}.bmp"
            cv2.imwrite(attacked_image_path, attacked_image)

            detected, wpsnr = detection.detection(
                original_image, watermarked_image, attacked_image_path
            )
            
            print(f"[Worker {worker_id}] Att: {param_str}, WPSNR: {wpsnr:.2f}, Det: {detected}")
            image_to_attack = attacked_image

            if detected == 0 and wpsnr >= 35:
                print(f"🎉 [Worker {worker_id}] Found a candidate! WPSNR: {wpsnr:.2f}")
                
                # --- MODIFIED: Put (wpsnr, path, history) into queue ---
                # We put WPSNR first so we can sort by it easily
                result_queue.put((wpsnr, attacked_image_path, attack_history))
                
                # We found a solution, so this "walk" is over.
                # Break the inner loop to start a new random walk.
                break
            
            # If quality drops too low, this "walk" is also over.
            if wpsnr <= 35:
                break


def run_parallel_search(
    original_image: str, 
    watermarked_image: str, 
    num_workers: int,
    search_duration_seconds: int
):
    """
    Launches workers and runs them for a fixed duration
    to find the best attack.
    """
    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # --- Mock files for the example (DELETE THIS) ---
    if not os.path.exists("input"): os.makedirs("input")
    if not os.path.exists(original_image):
        cv2.imwrite(original_image, np.random.randint(0, 256, (512, 512), dtype=np.uint8))
    if not os.path.exists(watermarked_image):
        cv2.imwrite(watermarked_image, np.random.randint(0, 256, (512, 512), dtype=np.uint8))
    # --- End of Mock ---
        
    print(f"Starting {num_workers} parallel workers...")
    print(f"Searching for the optimal attack for {search_duration_seconds} seconds.")
    
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        processes = []
        
        for i in range(num_workers):
            p = mp.Process(
                target=attack_worker,
                args=(i, original_image, watermarked_image, result_queue)
            )
            processes.append(p)
            p.start()

        # Let the workers run for the specified duration
        start_time = time.time()
        while (time.time() - start_time) < search_duration_seconds:
            time.sleep(1)
            print(f"Searching... {int(time.time() - start_time)}s / {search_duration_seconds}s")
        
        print("\n--- TIME'S UP ---")
        print("Terminating workers and collecting results...")

        # --- MODIFIED: Terminate all processes ---
        for p in processes:
            if p.is_alive():
                p.terminate() # Forcefully stop the worker
                p.join()      # Wait for it to shut down
        
        print("All workers stopped.")

        # --- NEW: Drain the queue and find the best result ---
        all_solutions = []
        while not result_queue.empty():
            try:
                all_solutions.append(result_queue.get_nowait())
            except queue.Empty:
                break
        
        if not all_solutions:
            print("\n--- OVERALL FAILURE ---")
            print("No solutions were found by any worker in the given time.")
            return

        print(f"\nFound {len(all_solutions)} total solutions. Finding the best...")
        
        # Sort the list by the first item (wpsnr) in descending order
        all_solutions.sort(key=lambda x: x[0], reverse=True)
        
        # The best one is the first item in the sorted list
        best_wpsnr, best_path, best_history = all_solutions[0]

        print("\n--- OPTIMAL ATTACK FOUND ---")
        print(f"Best WPSNR: {best_wpsnr:.2f}")
        print(f"Image saved to: {best_path}")
        print("Optimal Attack Sequence:")
        for i, step in enumerate(best_history):
            print(f"  Step {i+1}: {step}")

if __name__ == "__main__":
    WORKERS = os.cpu_count()
    if WORKERS is None:
        WORKERS = 4
        
    SEARCH_DURATION_SECONDS = 60 # Run for 1 minute
    
    run_parallel_search(
        "input/0036.bmp", 
        "output/0036_watermarked.bmp", 
        WORKERS,
        SEARCH_DURATION_SECONDS
    )
