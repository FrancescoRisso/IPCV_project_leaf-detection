from __future__ import annotations
from cv2.typing import MatLike
import cv2

from enum import Enum


class Formats(Enum):
    BGR = "BGR"
    HSV = "HSV"
    monochromatic = "monochromatic"


# FC[a][b] gives the value to give to cv2.cvtColor to convert from a to b
FORMAT_CONVERTER: dict[Formats, dict[Formats, int]] = {
    Formats.BGR: {
        Formats.HSV: cv2.COLOR_BGR2HSV,
        Formats.monochromatic: cv2.COLOR_BGR2GRAY,
    },
    Formats.HSV: {Formats.BGR: cv2.COLOR_HSV2BGR},
    Formats.monochromatic: {Formats.BGR: cv2.COLOR_GRAY2BGR},
}


class Image:
    def __init__(self, matrix: MatLike, format: Formats) -> None:
        """
        Creates a new image from a pre-existing matrix

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - matrix: the grid of pixels of the image
        - format: the color format of the matrix
        """
        self.__image = matrix
        self.__img_format = format

    @classmethod
    def fromFile(cls, path: str) -> Image:
        """
        Creates a new image, loading it from file.

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: the path where to open the image

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The Image
        """

        return cls(cv2.imread(path), Formats.BGR)

    def __convert_format(self, to: Formats) -> None:
        """
        Internally converts the format of the image, if it is not already in
        the target format

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - to: the requqested format
        """

        if self.__img_format == to:
            return

        try:
            converter = FORMAT_CONVERTER[self.__img_format][to]
        except Exception:
            raise Exception(f"Could not convert from {self.__img_format} to {to}")

        self.__image = cv2.cvtColor(self.__image, converter)

    def as_BGR(self) -> MatLike:
        """
        Returns the image as matrix of RGB values

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.BGR)
        return self.__image

    def as_HSV(self) -> MatLike:
        """
        Returns the image as matrix of HSV values

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.HSV)
        return self.__image

    def as_monochromatic(self) -> MatLike:
        """
        If the image is already saved in a monochromatic way*, it is simply
        returned as a matrix.
        Otherwhise, it is returned in grayscale.

        *An image can be monochromatic either because it has been converted
        to grayscale, or because it has been created as a single layer, for
        example by creating an image with only the hue layer

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.monochromatic)
        return self.__image

    def hue(self) -> MatLike:
        """
        Returns the image as matrix of hue values

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.HSV)
        return self.__image[:, :, 0]

    def saturation(self) -> MatLike:
        """
        Returns the image as matrix of saturation values

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.HSV)
        return self.__image[:, :, 1]

    def value(self) -> MatLike:
        """
        Returns the image as matrix of HSV value values

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrix of the image
        """

        self.__convert_format(Formats.HSV)
        return self.__image[:, :, 2]

    def hue_saturation_value(self) -> tuple[MatLike, MatLike, MatLike]:
        """
        Returns the image as three distinct matrices of hue, saturation,
        value

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The matrices of the image, as (hue, saturation, value)
        """

        self.__convert_format(Formats.HSV)
        return (self.__image[:, :, 0], self.__image[:, :, 1], self.__image[:, :, 2])

    def rotate_clockwise_90(self) -> None:
        """
        Rotates the image by 90° in the clockwise direction
        """
        self.__image = cv2.rotate(self.__image, cv2.ROTATE_90_CLOCKWISE)

    def rotate_anticlockwise_90(self) -> None:
        """
        Rotates the image by 90° in the anticlockwise direction
        """
        self.__image = cv2.rotate(self.__image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def save_as_file(self, path: str) -> None:
        """
        Stores the image as a file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: where to save the image (with the extension)
        """

        cv2.imwrite(path, self.__image)

    def dilate(self, kernel: MatLike, iterations: int = 1) -> None:
        """
        Performs the dilation morphological operator on the image
    
        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - kernel: the kernel to use on the operation
        - iterations: how many times the operation should be performed
        """
        for _ in range(iterations):
            self.__image = cv2.dilate(self.__image, kernel)

    def erode(self, kernel: MatLike, iterations: int = 1) -> None:
        """
        Performs the erosion morphological operator on the image
    
        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - kernel: the kernel to use on the operation
        - iterations: how many times the operation should be performed
        """
        for _ in range(iterations):
            self.__image = cv2.erode(self.__image, kernel)

    def close(self, kernel: MatLike, iterations: int = 1) -> None:
        """
        Performs the closure morphological operator on the image
    
        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - kernel: the kernel to use on the operation
        - iterations: how many times the operation should be performed
        """
        self.dilate(kernel, iterations)
        self.erode(kernel, iterations)

    def open(self, kernel: MatLike, iterations: int = 1) -> None:
        """
        Performs the opening morphological operator on the image
    
        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - kernel: the kernel to use on the operation
        - iterations: how many times the operation should be performed
        """
        self.erode(kernel, iterations)
        self.dilate(kernel, iterations)
