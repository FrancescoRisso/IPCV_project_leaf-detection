from cv2.typing import MatLike
import cv2

from functions.utils.rectangle import Rectangle
from functions.utils.segment import Segment
from functions.utils.leaf import is_px_leaf

from functions.lengths.leaf_height import find_leaf_height

from typing import TypeVar, List


T = TypeVar("T")
type tuple_of_11[T] = tuple[T, T, T, T, T, T, T, T, T, T, T]


def to_tuple_of_11(array: List[T]) -> tuple_of_11[T]:
    """
    Equivalent of python's tuple() function, to convert an array of 11
    items into a tuple_of_11

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - array: the array to be converted

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The tuple, if the array had 11 elements. 
    Otherwhise, an error is raised
    """
    if len(array) != 11:
        raise ValueError("Array must have exactly 11 elements")
    return (
        array[0],
        array[1],
        array[2],
        array[3],
        array[4],
        array[5],
        array[6],
        array[7],
        array[8],
        array[9],
        array[10],
    )


def __get_leaf_at_px(img: MatLike, paper_roi: Rectangle, row: int) -> Segment:
    """
    Returns the segment that contains the leaf at a given px height of
    the image

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image to be analyzed, in HSV
    - paper_roi: the region where there are only paper and leaf
    - row: the image row that should be considered

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The segment that includes the leaf
    """

    mincol = paper_roi.get_horiz().corner
    maxcol = paper_roi.get_horiz().other_corner()

    for col in range(mincol, maxcol):
        if is_px_leaf(img[row, col]):
            corner = col
            break

    for col in range(maxcol - 1, mincol, -1):
        if is_px_leaf(img[row, col]):
            other_corner = col
            break

    return Segment(corner, other_corner - corner)


def get_leaf_widths(
    img: MatLike, paper_roi: Rectangle, leaf_height: Segment | None = None
) -> tuple_of_11[Segment]:
    """
    Measures the width of the leaf every 10% of height (including 0% and
    100%)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image to be analyzed, in BGR
    - paper_roi: the region where there are only paper and leaf
    - leaf_height: if available, the segment that describes the leaf
        height. If it is not given, it is computed from scratch (a waste,
        if it was already available)

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A tuple of 11 segments, where the i-th element is the segment that
    represents the width and position of the leaf at 10*i% the height
    """

    segments = []

    leaf_height_certain = (
        find_leaf_height(img, paper_roi) if leaf_height is None else leaf_height
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # for index in range(0, 1):
    for index in range(0, 11):
        fraction = index * 1.0 / 10
        row = int(leaf_height_certain.corner + fraction * leaf_height_certain.length)
        segments.append(__get_leaf_at_px(img, paper_roi, row))

    return to_tuple_of_11(segments)