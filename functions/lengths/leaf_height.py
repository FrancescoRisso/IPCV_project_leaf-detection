from cv2.typing import MatLike
import cv2

from functions.utils.segment import Segment
from functions.utils.rectangle import Rectangle


MIN_LEAF_HUE = 0
MAX_LEAF_HUE = 80
MIN_LEAF_SAT = 110
MAX_LEAF_VAL = 150


def __is_leaf_in_line(img: MatLike, horiz_segment: Segment, y_coord: int) -> bool:
    """
    Checks if a line of pixels contains the leaf
    ---------------------------------------------------------------------

    Parameters
    ----------
    - img: the image to consider, in HSV
    - horiz_segment: the horizontal ROI where to look for
    - y_coord: the line of pixels to be analyzed

    ---------------------------------------------------------------------
    Returns
    -------
    Whether the chosen line includes parts of the leaf or not
    """

    for x in range(horiz_segment.corner, horiz_segment.other_corner()):
        (hue, sat, val) = img[y_coord, x]

        if (
            hue >= MIN_LEAF_HUE
            and hue <= MAX_LEAF_HUE
            and sat >= MIN_LEAF_SAT
            and val <= MAX_LEAF_VAL
        ):
            return True

    return False


def __find_leaf_extreme_recurs(
    img: MatLike, region: Rectangle, top_border: bool
) -> int:
    """
    Recursively finds the highest or lowest y level where there is the
    leaf, in a binary way
    ---------------------------------------------------------------------

    Parameters
    ----------
    - img: the image to consider, in HSV
    - region: the paper region, where to search
    - top_border: whether to look for the topmost (True) or bottommost
        (False) point of the leaf

    ---------------------------------------------------------------------
    Returns
    -------
    The y coordinate of the requested point
    """

    middle = region.vert.middle()
    leaf_present_in_middle = __is_leaf_in_line(img, region.get_horiz(), middle)

    # If the region to search is 1-tall, it's the end of the search
    if region.vert.length == 1:
        offset = 1 if top_border else -1
        return middle if leaf_present_in_middle else middle + offset

    if (leaf_present_in_middle and top_border) or (
        not leaf_present_in_middle and not top_border
    ):
        vert = region.vert.first_half()
    else:
        vert = region.vert.second_half()

    res = __find_leaf_extreme_recurs(
        img, Rectangle(region.get_horiz(), vert), top_border
    )

    if (region.get_vert().length < 0.05 * img.shape[0]) or __is_leaf_in_line(
        img, region.get_horiz(), res
    ):
        return res

    vert = region.get_vert().other_half(vert)

    return __find_leaf_extreme_recurs(
        img, Rectangle(region.get_horiz(), vert), top_border
    )


def find_leaf_height(img: MatLike, region: Rectangle) -> Segment:
    """
    Performs a binary search along the height of the image to find the y
    coordinates of the first and last px that includes the leaf.

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to consider, in BGR
    - region: the paper region, where to search

    ---------------------------------------------------------------------
    Returns
    -------
    The vertical segment where the leaf is present (with coordinates
    relative to the full image)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    top = __find_leaf_extreme_recurs(img, region, True)
    bottom = __find_leaf_extreme_recurs(img, region, False)

    return Segment(top, bottom - top)
