import numpy as np
import cv2
from scipy.fftpack import dct, idct
from sklearn.metrics import accuracy_score


# =========================================================
# === Utility functions ===================================
# =========================================================

def _dct2(img):
    return dct(dct(img.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

def _idct2(coeffs):
    return idct(idct(coeffs, axis=0, norm='ortho'), axis=1, norm='ortho')

def wpsnr(orig, proc):
    """Approximate PSNR, sufficiente per valutare la qualità."""
    mse = np.mean((orig.astype(np.float64) - proc.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# =========================================================
# === Embedding function ==================================
# =========================================================

def embed_mid_dct_cv2(image_path, key=2025, alpha=0.1,
                      low_cut=0.20, high_cut=0.60,
                      n_coefs_per_bit=128):
    """
    Embedding watermark di 1024 bit in bande medie DCT.
    Usa cv2 per I/O e salva il watermark in mark.npy.
    """
    # --- Lettura immagine ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile aprire {image_path}")
    rows, cols = img.shape

    # --- Creazione watermark (1024 bit) ---
    mark_size = 1024
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)
    print(f"Watermark generato e salvato: {mark_size} bit")

    # --- DCT ---
    ori_dct = _dct2(img)

    # --- Maschera per frequenze medie ---
    u = np.arange(rows)
    v_axis = np.arange(cols)
    U, V = np.meshgrid(u, v_axis, indexing='ij')
    dist = np.sqrt(U**2 + V**2)
    dist_norm = dist / dist.max()
    mask = (dist_norm >= low_cut) & (dist_norm <= high_cut)

    avail_idxs = np.argwhere(mask)
    mags = np.abs(ori_dct[mask])
    order = np.argsort(-mags)
    ordered_idxs = avail_idxs[order]

    rng_global = np.random.RandomState(key)
    rng_global.shuffle(ordered_idxs)

    total_needed = n_coefs_per_bit * mark_size
    if len(ordered_idxs) < total_needed:
        raise ValueError("Immagine troppo piccola per banda e parametri scelti")

    # --- Embedding spread-spectrum ---
    ptr = 0
    for b_idx, bit in enumerate(mark):
        sel = ordered_idxs[ptr: ptr + n_coefs_per_bit]
        ptr += n_coefs_per_bit

        rng = np.random.RandomState(key + 1000 + b_idx)
        pn = rng.normal(0, 1, n_coefs_per_bit)
        sign = 1.0 if bit == 1 else -1.0

        for (i, j), noise in zip(sel, pn):
            ori_dct[i, j] += alpha * sign * noise

    # --- IDCT ---
    wm_img = _idct2(ori_dct)
    wm_img = np.clip(np.rint(wm_img), 0, 255).astype(np.uint8)

    out_path = "watermarked.png"
    cv2.imwrite(out_path, wm_img)
    print(f"Immagine watermarked salvata in: {out_path}")
    print(f"WPSNR: {wpsnr(img, wm_img):.2f} dB")

    meta = {
        'key': key,
        'alpha': alpha,
        'low_cut': low_cut,
        'high_cut': high_cut,
        'n_coefs_per_bit': n_coefs_per_bit,
        'mark_size': mark_size
    }
    return wm_img, img, meta


# =========================================================
# === Detection function ==================================
# =========================================================

def detect_mid_dct_cv2(wm_image_path, meta):
    """
    Estrae watermark dalla sola immagine watermarked (blind detection).
    """
    img_wm = cv2.imread(wm_image_path, cv2.IMREAD_GRAYSCALE)
    if img_wm is None:
        raise FileNotFoundError(f"Impossibile aprire {wm_image_path}")

    rows, cols = img_wm.shape
    coeffs = _dct2(img_wm)

    u = np.arange(rows)
    v_axis = np.arange(cols)
    U, V = np.meshgrid(u, v_axis, indexing='ij')
    dist = np.sqrt(U**2 + V**2)
    dist_norm = dist / dist.max()
    mask = (dist_norm >= meta['low_cut']) & (dist_norm <= meta['high_cut'])
    avail_idxs = np.argwhere(mask)

    mags = np.abs(coeffs[mask])
    order = np.argsort(-mags)
    ordered_idxs = avail_idxs[order]

    rng_global = np.random.RandomState(meta['key'])
    rng_global.shuffle(ordered_idxs)

    mark_length = meta['mark_size']
    n_coefs_per_bit = meta['n_coefs_per_bit']

    detected_bits = []
    pos = 0
    for b_idx in range(mark_length):
        sel = ordered_idxs[pos: pos + n_coefs_per_bit]
        pos += n_coefs_per_bit
        rng = np.random.RandomState(meta['key'] + 1000 + b_idx)
        pn = rng.normal(0, 1, n_coefs_per_bit)
        vals = np.array([coeffs[int(i), int(j)] for (i, j) in sel])
        corr = np.dot(vals, pn)
        detected_bits.append(1 if corr > 0 else 0)

    detected_bits = np.array(detected_bits, dtype=np.uint8)
    np.save("detected.npy", detected_bits)
    print("Watermark estratto e salvato in detected.npy")
    return detected_bits


# =========================================================
# === ESEMPIO USO =========================================
# =========================================================

if __name__ == "__main__":
    # Inserisci il percorso di un'immagine (grayscale o color)
    image_path = "images/0001.bmp"  # <-- cambia qui

    # EMBEDDING
    wm_img, orig_img, meta = embed_mid_dct_cv2(image_path, key=2025, alpha=0.5)
    print("Embedding completato.")
    print()
    # DETECTION
    detected = detect_mid_dct_cv2("watermarked.png", meta)

    # Confronto watermark originali
    mark = np.load("mark.npy")
    acc = accuracy_score(mark, detected)
    print(f"Accuracy watermark rilevato: {acc*100:.2f}%")