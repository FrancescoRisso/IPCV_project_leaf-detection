import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from typing import Tuple
from functions.utils.leaf import get_leaf_mask
from functions.lengths.paper_roi import find_roi_boundaries

img = cv2.imread(os.path.join(base_path, foglia))
l, r, t, b = find_roi_boundaries(img)
img = img[t:b, l:r]
mask = get_leaf_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

def find_leaf_contour(mask: MatLike) -> np.ndarray[Any, dtype[generic]]:
    """
    The function retrives the leaf contour using openCV findContours
    function
    ---------------------------------------------------------------------

    PARAMETERS
    ----------
    - leafMask: the tresholded image of the leaf, a bitmap

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The openCV contour of the leaf (the main one), as a numpy array.
    It is accepted by the openCV functions which work with contours 
    """

    # a huge closing operation takes place, using a circular kernel. It is a necessary
    # passage, all the holes in the leaves are filled to avoid problems
    # in distinguishing between different contours
    ker2 = np.ones((6,6), np.uint8)
    ker3 = np.ones((2,2), np.uint8)
    kerR  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(104, 104))

    mask = cv2.dilate(mask, kerR)
    mask = cv2.erode(mask, kerR)

    # noise cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((12,12)))

    # add a 10px black border around the image, so the leaves which exceed the image
    # dimensions are properly elaborated by Canny and then by the findContour function
    mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # N.B. 
    # the new coordinates are not usable on the leaf maske beacuse now the image is bigger,
    # however the features which will be extracted are not position related so we'll ignore that 

    edge = cv2.Canny(mask, 175, 175)
    edge = cv2.dilate(edge, ker2)   
    edge = cv2.erode(edge, ker2)
    edge = cv2.dilate(edge, ker3)  

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours contains all the detected contours

    # if many contours are detected, we select the right one by finding the biggest, 
    # so the one which contains the most pixels
    bestCnt = None
    if len(contours) > 1 :
        bestArea = 0
        for cont in contours:
            areaC = cv2.contourArea(cont)
            if areaC > bestArea:
                bestCnt = cont
        contour = bestCnt
    else:
        contour = contours[0]

    return contour


def get_leaf_perimeter(contour: np.array) -> float:
    """
    The function retrives the length of the leaf perimeter
    ---------------------------------------------------------------------

    PARAMETERS
    ----------
    - contour: the OpenCV contour of the leaf, a numpy array

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The length of the perimeter, as a floating point number 
    """

    perimeter = cv2.arcLength(contour, False)

    return perimeter


def get_leaf_convexity(contour: np.array) -> float:
    """
    The function computes the convexity of the leaf
    ---------------------------------------------------------------------

    PARAMETERS
    ----------
    - contour: the OpenCV contour of the leaf, a numpy array

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A floating point value representing the convexity of the leaf.
    Returns the difference between the convex hull area and the leaf area.
    The more convex teh leaf, the smaller is the number. 
    """

    hull = cv2.convexHull(contour)

    areaContour = cv2.contourArea(contour)

    areaHull = cv2.contourArea(hull)

    return (areaHull-areaContour)
