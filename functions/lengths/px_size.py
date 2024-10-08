import cv2
import numpy as np

from cv2.typing import MatLike

from functions.utils.rectangle import Rectangle
from functions.utils.image import crop_image
from functions.utils.leaf import get_leaf_mask

from functions.lengths.px_counting import count_paper_pixels


A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def __max_sat_min_val(img: MatLike, paper_roi: Rectangle) -> tuple[int, int]:
    """
    Computes approximately the maximum saturation and minimum value of
    pixels of paper

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the full image, in HSV
    - paper_roi: a region where only paper and leaf exist

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A tuple of (max saturation, min value)
    """

    # Crop image to a region with only paper and leaf
    img = crop_image(img, paper_roi)

    # Compute the leaf and inverse (=paper) mask
    leaf = get_leaf_mask(img)
    paper = cv2.bitwise_not(leaf)

    # Remove the lighter area around the leaf
    erosion_size = paper_roi.horiz.length // 10
    paper = cv2.erode(paper, np.ones((erosion_size, erosion_size)))
    leaf = cv2.dilate(leaf, np.ones((erosion_size, erosion_size)))

    # Remove the leaf from the saturation channel
    sat = img[:, :, 1]
    masked_sat = cv2.bitwise_and(sat, paper)

    # Remove the leaf from the value channel
    val = img[:, :, 2]
    masked_val = cv2.bitwise_or(val, leaf)
    # cv2.imwrite("./test/valmask.jpg", masked_val)

    # Return the max saturation
    return (int(masked_sat.max()), int(masked_val.min()))


def get_px_size(img: MatLike, paper_roi: Rectangle, height: bool) -> float:
    """
    Returns a size (width or height) in mm of a pixel of the picture,
    obtained by comparing the A4 paper sizes (in mm) to the number of
    pixels of paper.

    The number of paper pixels is measured as a median of 5 different
    rows/columns around the middle of the image.

    ---------------------------------------------------------------------
    Parameters
    ----------
    - img: the image to be considered, in HSV
    - paper_roi: a region where there are only paper and leaf
    - height: whether the function should compute the height or width of
        a px

    ---------------------------------------------------------------------
    Returns
    -------
    The average width in mm of a pixel of the picture
    """
    # Size in px of the img side orthogonal to the direction we are measuring
    orthogonal_size = img.shape[1] if height else img.shape[0]

    (max_paper_sat, min_paper_val) = __max_sat_min_val(img, paper_roi)

    lengths = []
    for frac in [0.4, 0.45, 0.5, 0.55, 0.6]:
        lengths.append(
            count_paper_pixels(
                img,
                int(orthogonal_size * frac),
                height,
                max_paper_sat,
                min_paper_val,
            ).length
        )

    # Use the median value
    lengths.sort()
    tot_size_px = lengths[len(lengths) // 2]

    tot_size_mm = A4_HEIGHT_MM if height else A4_WIDTH_MM

    return tot_size_mm * 1.0 / tot_size_px
