from functions.utils.segment import Segment

import cv2
from cv2.typing import MatLike


WHITE_THRESHOLD = 35
CONSECUTIVE_PX_OF_PAPER = 20
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def get_px_height_in_mm(img: MatLike) -> float:
    """
    Returns the height in cm of a pixel of the picture, obtained by
    comparing the A4 paper height (in cm) to the number of pixels of
    paper in the central column of pixels

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered

    ---------------------------------------------------------------------
    Returns
    -------
    The average height in cm of a pixel of the picture
    """

    w = img.shape[1]

    paper_height: int = __count_paper_pixels_at_col(img, int(0.5 * w)).length

    return A4_HEIGHT_MM * 1.0 / paper_height


def get_px_width_in_mm(img: MatLike) -> float:
    """
    Returns the width in cm of a pixel of the picture, obtained by
    comparing the A4 paper width (in cm) to the number of pixels of
    paper in the central row of pixels

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered

    ---------------------------------------------------------------------
    Returns
    -------
    The average width in cm of a pixel of the picture
    """

    h = img.shape[0]

    paper_width: int = __count_paper_pixels_at_row(img, int(0.5 * h)).length

    return A4_WIDTH_MM * 1.0 / paper_width


def __count_paper_pixels_at_row(img: MatLike, row_no: int) -> Segment:
    """
    Checks which pixels of a given row of pixels of an image are part of
    a white paper sheet.

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered, as HSV
    - row_no: the index of the row to be considered

    ---------------------------------------------------------------------
    Returns
    The distance of the paper sheet from the left margin (called d) and
    the width w of the sheet, in a tuple structured as (d, w), with all
    measures in pixels
    """
    row = img[row_no : row_no + 1, :]
    return __count_paper_pixels(row)


def __count_paper_pixels_at_col(img: MatLike, col_no: int) -> Segment:
    """
    Checks which pixels of a given colum of pixels of an image are part of
    a white paper sheet.

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered
    - col_no: the index of the column to be considered

    ---------------------------------------------------------------------
    Returns
    The distance of the paper sheet from the top margin (called d) and
    the height h of the sheet, in a tuple structured as (d, h), with all
    measures in pixels
    """
    col = img[:, col_no : col_no + 1, :]
    col = cv2.rotate(col, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return __count_paper_pixels(col)


def __count_paper_pixels(img: MatLike) -> Segment:
    """
    Checks which pixels of a one-row image are part of a white paper
    sheet

    This is performed by analyzing the first row of the image, counting
    what is not white at the left and at the right of the image.
    ---------------------------------------------------------------------

    Parameters
    ----------
    - img: the row of the image to be considered

    ---------------------------------------------------------------------
    Returns
    The segment that describes the paper sheet in the middle of the image
    """

    w = img.shape[1]
    saturation = img[:, :, 1]

    margin_left: int = 0
    margin_right: int = 0

    # CONSECUTIVE_PX_OF_PAPER white pixels are required to reduce noise
    # This value is then subtracted by the count
    consecutive_match: int = 0

    # from the left
    for delta in range(w):
        if saturation[0, 0 + delta] < WHITE_THRESHOLD:
            consecutive_match += 1

            if consecutive_match == CONSECUTIVE_PX_OF_PAPER:
                margin_left += delta - consecutive_match
                consecutive_match = 0
                break
        else:
            consecutive_match = 0

    # from the right
    for delta in range(w):
        if saturation[0, w - delta - 1] < WHITE_THRESHOLD:
            consecutive_match += 1

            if consecutive_match == CONSECUTIVE_PX_OF_PAPER:
                margin_right += delta - consecutive_match
                consecutive_match = 0
                break
        else:
            consecutive_match = 0

    return Segment(margin_left, w - margin_left - margin_right)
