from functions.lengths.px_counting import (
    count_paper_pixels_at_col,
    count_paper_pixels_at_row,
)

import cv2
from cv2.typing import MatLike


A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# TODO: change count functions to more robust ones


def get_px_height_in_mm(img: MatLike) -> float:
    """
    Returns the height in mm of a pixel of the picture, obtained by
    comparing the A4 paper height (in mm) to the number of pixels of
    paper in the central column of pixels

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered, in HSV

    ---------------------------------------------------------------------
    Returns
    -------
    The average height in mm of a pixel of the picture
    """

    h = img.shape[0]
    w = img.shape[1]

    paper_height: int = count_paper_pixels_at_col(img, int(0.5 * w)).length

    res = A4_HEIGHT_MM * 1.0 / paper_height
    max_reasonable = A4_HEIGHT_MM * 2.0 / h

    # A reasonable maximum is the result that would be obtained when the
    # paper only occupies half of the picture

    if res > max_reasonable:
        # there is something wrong, reduce saturation and re-evaluate
        img[:, :, 1] = img[:, :, 1] / 2  # type: ignore
        return get_px_height_in_mm(img)

    return res


def get_px_width_in_mm(img: MatLike) -> float:
    """
    Returns the width in mm of a pixel of the picture, obtained by
    comparing the A4 paper width (in mm) to the number of pixels of
    paper in the central row of pixels

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered, in HSV

    ---------------------------------------------------------------------
    Returns
    -------
    The average width in mm of a pixel of the picture
    """

    h = img.shape[0]
    w = img.shape[1]

    paper_width: int = count_paper_pixels_at_row(img, int(0.5 * h)).length

    res = A4_WIDTH_MM * 1.0 / paper_width
    max_reasonable = A4_WIDTH_MM * 2.0 / w

    # A reasonable maximum is the result that would be obtained when the
    # paper only occupies half of the picture

    if res > max_reasonable:
        # there is something wrong, reduce saturation and re-evaluate
        img[:, :, 1] = img[:, :, 1] / 2  # type: ignore
        return get_px_width_in_mm(img)

    return res
