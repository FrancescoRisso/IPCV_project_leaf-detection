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


def draw_rectangle(
    img: MatLike, rect: Rectangle, color: tuple[int, int, int], thickness: int = 1
) -> MatLike:
    """
    Draws a rectangle on top of the image

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - img: the original image
    - rect: the rectangle to draw on top
    - color: the color of the rectangle, with the same color format as
        the image
    - thickness: how many pixels should the rectangle border be thick

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The image with the rectangle on top
    """

    corner1 = (rect.get_horiz().corner, rect.get_vert().corner)
    corner2 = (
        rect.get_horiz().corner + rect.get_horiz().length,
        rect.get_vert().corner + rect.get_vert().length,
    )
    return cv2.rectangle(img, corner1, corner2, color, thickness)