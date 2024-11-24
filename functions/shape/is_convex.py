from cv2.typing import MatLike
import cv2

from custom_types.tuple_of_11 import tuple_of_11

from functions.utils.segment import Segment
from functions.utils.leaf import get_leaf_mask


TESTS = 21



def is_likely_convex(
    img: MatLike, leaf_max_width: Segment, leaf_height: Segment
) -> bool:
    """
    Checks if a leaf is "likely convex". The covexity is tested at
    samples of leaf height.

    If the function returns false, then a concavity point is found and
    the leaf is clearly concave.

    If instead the function returns true, the samples did not have any
    concavity points, therefore the leaf is either convex, or has
    extremely small concavities.

    The convexity is tested at TESTS equispaced pixels rows (including
    the top and bottom one: for example, TESTS = 21 means testing at 0%,
    5%, 10%, ... 95%, 100% of the height)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image to be analyzed, in BGR
    - leaf_max_width: the maximum leaf width, as a segment
    - leaf_height: the segment that describes the leaf height

    ---------------------------------------------------------------------
    OUTPUT
    ------
    - false, if the leaf is concave
    - true, if the leaf is either convex or has small, undetected
        concavities
    """

    mask = get_leaf_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    for index in range(0, TESTS):
        fraction = index * 1.0 / (TESTS - 1)
        row = int(leaf_height.corner + fraction * leaf_height.length)

        # If the row is not convex, the leaf is not convex
        if not __is_row_convex(mask, row, leaf_max_width):
            return False

    return True


def __is_row_convex(img: MatLike, row: int, leaf_segment: Segment) -> bool:
    """
    Checks if a row of pixels of leaf is convex (from when the leaf
    starts to when the leaf ends, there are no air gaps)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the bit-mask of the leaf image
    - row: the row of px to analyze
    - leaf_segment: the extremes of the leaf

    ---------------------------------------------------------------------
    OUTPUT
    ------
    If the selected row of pixels is convex
    """
    start = 0
    end = 0

    for col in range(leaf_segment.corner, leaf_segment.other_corner()):
        if img[row, col]:
            start = col
            break

    for col in range(leaf_segment.other_corner(), leaf_segment.corner, -1):
        if img[row, col]:
            end = col
            break

    for col in range(start, end):
        # If px is not leaf, the leaf is not convex
        if not img[row, col]:
            return False

    return True
