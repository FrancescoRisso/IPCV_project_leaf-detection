from functions.utils.segment import Segment
from functions.utils.rectangle import Rectangle

from cv2.typing import MatLike

import cv2
import numpy as np


def count_paper_pixels(
    img: MatLike,
    level: int,
    vert: bool,
    max_paper_sat: int,
    min_paper_val: int,
) -> Segment:
    """
    Checks which pixels of a row/col of an image are part of a white
    paper sheet.

    This is performed by counting what is not white at the left/right or
    at the top/bottom of the selected row/col of the image.

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the row of the image to be considered, in HSV
    - level: the row/column to be evaluated
    - vert: if the function should count the paper pixels in a column
        (vert=True, level=col) or in a row (vert=False, level=row)
    - max_paper_sat: an approximate value of the maximum saturation of
        the paper pixels
    - min_paper_val: an approximate value of the maximum value of the
        paper pixels

    ---------------------------------------------------------------------
    Returns
    The segment that describes the paper sheet in the middle of the image
    """

	# Consider only saturations < max_paper_sat, with an opening on that to remove noise
    saturation = img[:, :, 1]
    saturation = cv2.threshold(saturation, max_paper_sat, 255, cv2.THRESH_BINARY_INV)[1]
    saturation = cv2.morphologyEx(saturation, cv2.MORPH_OPEN, np.ones((51, 51)))

	# Consider only values > min_paper_val, with an opening on that to remove noise
    value = img[:, :, 2]
    value = cv2.threshold(value, min_paper_val, 255, cv2.THRESH_BINARY)[1]
    value = cv2.morphologyEx(value, cv2.MORPH_OPEN, np.ones((51, 51)))

	# AND the two, to have a mask of where the paper is
    img = cv2.bitwise_and(saturation, value)

	# If the request is to compute the vertical paper size, rotate the image
    # to compute the horizontal paper size and obtain the same value
    if vert:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    w = img.shape[1]

    margin_left: int = 0
    margin_right: int = 0

    # Measure margin from the left
    for delta in range(w):
        if img[level, 0 + delta]:
            margin_left = delta
            break

    # Measure margin from the right
    for delta in range(w):
        if img[level, w - delta - 1]:
            margin_right = delta
            break

    return Segment(margin_left, w - margin_left - margin_right)
