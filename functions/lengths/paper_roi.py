import cv2
import numpy as np
from cv2.typing import MatLike
from typing import Tuple

from functions.utils.rectangle import Rectangle

WHITE_THRESHOLD = 120
NUM_OF_SAMPLES = 30


def __detect_lines(img: MatLike) -> Tuple[MatLike, MatLike]:
    """
    The function uses the Hough transform to detect the border of the
    paper sheet

    ---------------------------------------------------------------------
    PARAMETERS
    ----------

    - img: the image, in BGR

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A tuple, composed of:
    - the list of segments, expressed like [x1 y1 x2 y1] (!)
        note that each element is still a list, of only one element(?),
        to access the points you must acces line[0] = [x1 y1 x2 y2]
    - the image thresholded with the WHITE_THRESHOLD value
    """

    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgG = cv2.blur(imgG, (8, 8))
    thImg = cv2.threshold(imgG, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    ker1 = np.ones((26, 26), np.uint8)
    thImg = cv2.erode(thImg, ker1)
    thImg = cv2.dilate(thImg, ker1)

    ker = np.ones((3, 3), np.uint8)

    contour = cv2.morphologyEx(thImg, cv2.MORPH_GRADIENT, ker)

    contour = cv2.dilate(contour, ker)

    # lines is a list of each line found expressed like [x1 y1 x2 y1] (!)

    lines = cv2.HoughLinesP(contour, 1, np.pi / 2, 50, minLineLength=150, maxLineGap=80)
    # ! note that line is still a list, of only one element (?), to access
    #   the points you must acces line[0] = [x1 y1 x2 y2]

    return lines, thImg


def __find_paper_margin(thImg: MatLike) -> Tuple[int, int, int, int]:
    """
    The function finds the 4 margins of the paper sheet, using a median
    value.
    From each side it checks NUM_OF_SAMPLES times the distance of the
    paper's white, then it returns the median value

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - thImg: the image thresholded with the WHITE_THRESHOLD value

    ---------------------------------------------------------------------
    OUTPUT
    ------
    - marginL, marginR, marginT, marginB are the pixel positions of the
      left, right, top and bottom margin of the paper sheet
    """

    imgH, imgW = thImg.shape[:2]
    marginL = 0
    marginR = imgW
    marginT = 0
    marginB = imgH

    # trovo il margine sinistro: partendo dal bordo immagine avanzo fino al foglio per più (NUM_OF_SAMPLES) volte
    # la coordinata x del margine sarà la mediana dei valori deltaX , cioè la mediana delle coordinate dei punti del bordo

    # !!! da gestire errore per accesso fuori dall'immagine

    # left border
    samples = []
    deltaY = 0
    for i in range(NUM_OF_SAMPLES):

        deltaX = 0
        while thImg[imgH // 5 + deltaY, 0 + deltaX] == 0:  # N.B. !!! img(y, x)
            deltaX += 1

        samples.append(deltaX)

        deltaY += int((3 / 5 * imgH) / NUM_OF_SAMPLES)

    samples.sort()
    marginL = samples[len(samples) // 2]
    samples.clear()

    # right border
    samples = []
    deltaY = 0
    for i in range(NUM_OF_SAMPLES):

        deltaX = imgW - 1
        while thImg[imgH // 5 + deltaY, 0 + deltaX] == 0:  # N.B. !!! img(y, x)
            deltaX -= 1

        samples.append(deltaX)

        deltaY += int((3 / 5 * imgH) / NUM_OF_SAMPLES)

    samples.sort()
    marginR = samples[len(samples) // 2]
    samples.clear()

    # top border
    samples = []
    deltaX = 0
    for i in range(NUM_OF_SAMPLES):

        deltaY = 0
        while thImg[0 + deltaY, imgW // 5 + deltaX] == 0:  # N.B. !!! img(y, x)
            deltaY += 1

        samples.append(deltaY)

        deltaX += int((3 / 5 * imgW) / NUM_OF_SAMPLES)

    samples.sort()
    marginT = samples[len(samples) // 2]
    samples.clear()

    # bottom border
    samples = []
    deltaX = 0
    for i in range(NUM_OF_SAMPLES):

        deltaY = imgH - 1
        while thImg[0 + deltaY, imgW // 5 + deltaX] == 0:  # N.B. !!! img(y, x)
            deltaY -= 1

        samples.append(deltaY)

        deltaX += int((3 / 5 * imgW) / NUM_OF_SAMPLES)

    samples.sort()
    marginB = samples[len(samples) // 2]
    samples.clear()

    return marginL, marginR, marginT, marginB


def find_roi_boundaries(img: MatLike) -> Tuple[int, int, int, int]:
    """
    The function finds the 4 pixel values of the paper sheet side,
    in such a way to extract a rectangular region of interest which
    inlcludes only white paper and the leaf.
    It iterates on the segments found by the function __detect_lines
    and it assign them to the relative paper margin. Then the most
    conservative value is chosen, so there will be no backruond
    in the extracted roi. The roi will be img[roiT:roiB , roiL:roiR]

    ---------------------------------------------------------------------
    PARAMETERS
    ----------

    - img: the image, in BGR

    ---------------------------------------------------------------------
    OUTPUT
    ------
    - roiL, roiR, roiT, roiB: piexl values of the 4 sides of the roi,
        the left, right, top and bottom one. A slice can be calculated
        like img[ roiT:roiB , roiL:roiR ]
    """

    # % of the min between height and width of the image that i want my roi to be reduced by
    PADDING = 0.9

    # constant percentual value, relative to the min beween the image width and height
    # it dictates if a segment belongs or not to a border, based on the distance
    DIST_PERC = 1.8

    imgH, imgW = img.shape[:2]
    roiL = 0
    roiR = imgW
    roiT = 0
    roiB = imgH

    lines, thImg = __detect_lines(img)

    marginL, marginR, marginT, marginB = __find_paper_margin(thImg)

    def isVertical(points: list[int]) -> int:
        if abs(points[2] - points[0]) < 20:
            return 1
        return 0

    def isHorizontal(points: list[int]) -> int:
        if abs(points[3] - points[1]) < 20:
            return 1
        return 0

    def belongsToSide(point: int, side: int) -> int:
        if (point < side + int(((DIST_PERC / 100) * min(imgW, imgH)))) and (
            point > side - int(((DIST_PERC / 100) * min(imgW, imgH)))
        ):
            return 1
        return 0

    for line in lines:
        line = line[0]
        if isVertical(line):

            # check if the segmemt belongs to the left border
            if belongsToSide(line[0], marginL) or belongsToSide(line[2], marginL):
                # if true I eventually update roiL with a more conservative value
                if max(line[0], line[2]) > roiL:
                    roiL = max(line[0], line[2])

            elif belongsToSide(line[0], marginR) or belongsToSide(line[2], marginR):
                if min(line[0], line[2]) < roiR:
                    roiR = min(line[0], line[2])

        elif isHorizontal(line):

            # check if the segment belongs to the top border
            if belongsToSide(line[1], marginT) or belongsToSide(line[3], marginT):
                # if it does i eventually update it with a more conservative value
                if max(line[1], line[3]) > roiT:
                    roiT = max(line[1], line[3])

            elif belongsToSide(line[1], marginB) or belongsToSide(line[3], marginB):
                if min(line[1], line[3]) < roiB:
                    roiB = min(line[1], line[3])

    # restirct the roi area with the specified padding
    padd = int(PADDING / 100 * min(imgH, imgW))
    roiL += padd
    roiR -= padd
    roiT += padd
    roiB -= padd

    return roiL, roiR, roiT, roiB


def roi_boundaries_as_rect(roi: tuple[int, int, int, int]) -> Rectangle:
    """
    Given the paper ROI as a tuple, returns it as a Rectangle

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - roi: the result of the function find_roi_boundaries

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The ROI as a Rectangle
    """

    return Rectangle.from_values(roi[2], roi[0], roi[1] - roi[0], roi[3] - roi[2])
