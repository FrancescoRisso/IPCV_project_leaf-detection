from cv2.typing import MatLike
import cv2

from functions.utils.rectangle import Rectangle
from functions.utils.image import crop_image
from functions.utils.leaf import get_leaf_mask


def get_avg_color(img: MatLike, leaf_roi: Rectangle) -> tuple[float, float, float]:
    """
    Returns the color obtained as the average color of all the leaf px

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the image, in BGR
    - leaf_roi: the region where the leaf is

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The average color, as a tuple (hue, saturation, value)
    """

    img = crop_image(img, leaf_roi)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = get_leaf_mask(img)

    avg = cv2.mean(img, mask)

    return (avg[0], avg[1], avg[2])
