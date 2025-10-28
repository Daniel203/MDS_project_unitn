import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_curve

from attacks import randomized_attack
from detection_ACME import extract_watermark, similarity
from embedding import embed_watermark

N_ATTACKS = 10
"""The number of randomized attacks to make per every image"""

MARK_SIZE = 1024
"""The size of the watermark. It's 1024 as defined in the challange constraints"""

ALPHA = 1.0
"""The alpha value used when embedding and decoding"""



def __get_labels_and_score_arrays(
    images: list[MatLike], watermark: NDArray[np.uint8]
) -> tuple[list[np.float64], list[int]]:
    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores: list[np.float64] = []
    labels: list[int] = []

    for image in images:
        for _ in range(N_ATTACKS):
            # Embed the watermark with real embedding
            params = [ALPHA, "additive"]
            image_watermarked = embed_watermark(image, watermark, params)

            # Fake watermark for H0
            watermark_fake = np.random.choice([-1.0, 1.0], MARK_SIZE)

            # Random attack
            image_attacked = randomized_attack(image_watermarked)

            # Extract attacked watermark
            watermark_attacked = extract_watermark(image, image_attacked, params)
            watermark_extracted = extract_watermark(image, image_watermarked, params)

            # Compute similarity H1
            scores.append(similarity(watermark_extracted, watermark_attacked))
            labels.append(1)

            # Compute similarity H0
            scores.append(similarity(watermark_fake, watermark_attacked))
            labels.append(0)

    return scores, labels


def __generate_plot_and_save_on_file(
    scores: list[np.float64], labels: list[int], filename: str = "roc_curve.png"
):
    # Compute ROC
    fpr, tpr, tau = roc_curve(
        np.asarray(labels), np.asarray(scores), drop_intermediate=False
    )

    # Compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # Save plot 
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()  # Free memory, close the figure

    # Find TPR and threshold near FPR = 0.05
    idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))

    print("For a FPR ≈ 0.05, TPR = %0.2f" % tpr[idx_tpr[0][0]])
    print("For a FPR ≈ 0.05, threshold = %0.2f" % tau[idx_tpr[0][0]])
    print("Check FPR %0.2f" % fpr[idx_tpr[0][0]])

    print(f"ROC curve saved to: {filename}")
    return


def roc(images: list[MatLike], watermark: NDArray[np.uint8]):
    scores, labels = __get_labels_and_score_arrays(images, watermark)
    __generate_plot_and_save_on_file(scores, labels)

    return
