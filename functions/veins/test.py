from cv2.typing import MatLike
import cv2
import matplotlib.pyplot as plt
import numpy as np

from functions.utils.rectangle import Rectangle
from functions.utils.image import crop_image
from functions.utils.leaf import get_leaf_mask


def extract_veins(img: MatLike, leaf_roi: Rectangle) -> None:
    """


    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    -

    ---------------------------------------------------------------------
    OUTPUT
    ------

    """

    leaf = crop_image(img, leaf_roi)
    mask = get_leaf_mask(leaf)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

    mask = cv2.erode(mask, np.ones((7,7)))

    leaf = cv2.bitwise_and(leaf, mask)

    smalls = {
        51: [11, 21, 41, 47],
        101: [15, 65, 85, 95],
        201: [51, 115, 169, 189],
        301: [91, 185, 239, 289]
    }

    for row, large in enumerate([51, 101]):
        blur_large = cv2.GaussianBlur(leaf, (large, large), 0)

        for col, small in enumerate(smalls[large]):
            blur_small = cv2.GaussianBlur(leaf, (small, small), 0)
            delta = cv2.subtract(blur_large, blur_small)

            plt.subplot(2, 4, (4 * row + col + 1))
            plt.imshow(delta, cmap="gray")
            plt.axis(False)
            plt.title(f"{small}, {large}")

    plt.show()
