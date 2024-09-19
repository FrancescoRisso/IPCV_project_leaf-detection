from cv2.typing import MatLike
import cv2

from functions.utils.rectangle import Rectangle


def crop_image(img: MatLike, roi: Rectangle) -> MatLike:
    """
    Crops an image to a specific rectangular region

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image to be cropped
    - roi: the rectangle of image to be kept

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The image, cropped to the specified region
    """
    return img[
        roi.get_vert().corner : roi.get_vert().other_corner(),
        roi.get_horiz().corner : roi.get_horiz().other_corner(),
        :,
    ]
