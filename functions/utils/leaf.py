from cv2.typing import MatLike
import cv2
import numpy as np


MIN_LEAF_HUE = 0
MAX_LEAF_HUE = 80
MIN_LEAF_SAT = 100
MAX_LEAF_VAL = 150


def is_px_leaf(px: tuple[int, int, int]) -> bool:
    """
    Looking at the colors, tells if a pixel can be part of a leaf or not

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - px: the pixel to analyse, described as HSV tuple

    ---------------------------------------------------------------------
    OUTPUT
    ------
    Whether the pixel can belong to a leaf or not
    """

    hue, sat, val = px

    if (
        hue >= MIN_LEAF_HUE
        and hue <= MAX_LEAF_HUE
        and sat >= MIN_LEAF_SAT
        and val <= MAX_LEAF_VAL
    ):
        return True

    return False


def get_leaf_mask(img: MatLike) -> MatLike:
    """
    Returns a mask to identify the exact region where the leaf is.
    It is done by first applying thresholds on the 3 channels, then the
    masks are and-ed, and finally a closing operation is executed to
    remove some noise inside the leaf

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image, in HSV

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The mask that represents the leaf
    """

    min_hue_mask = cv2.threshold(img[:, :, 0], MIN_LEAF_HUE, 180, cv2.THRESH_BINARY)[1]
    res = min_hue_mask

    max_hue_mask = cv2.threshold(
        img[:, :, 0], MAX_LEAF_HUE, 180, cv2.THRESH_BINARY_INV
    )[1]
    res = cv2.bitwise_and(res, max_hue_mask)

    min_sat_mask = cv2.threshold(img[:, :, 1], MIN_LEAF_SAT, 255, cv2.THRESH_BINARY)[1]
    res = cv2.bitwise_and(res, min_sat_mask)

    max_val_mask = cv2.threshold(
        img[:, :, 2], MAX_LEAF_VAL, 255, cv2.THRESH_BINARY_INV
    )[1]
    res = cv2.bitwise_and(res, max_val_mask)

    return cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((21,21)))
