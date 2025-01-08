import cv2
import numpy as np

from cv2.typing import MatLike

def find_leaf_contour(mask: MatLike) -> MatLike:
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
    The openCV contour of the leaf (the main one).
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
    maskEx = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, dst=None, value=[0, 0, 0])

    # N.B. 
    # the new coordinates are not usable on the leaf maske beacuse now the image is bigger,
    # however the features which will be extracted are not position related so we'll ignore that 

    edge = cv2.Canny(maskEx, 175, 175)
    edge = cv2.dilate(edge, ker2)   
    edge = cv2.erode(edge, ker2)
    edge = cv2.dilate(edge, ker3)  

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours contains all the detected contours

    # if many contours are detected, we select the right one by finding the biggest, 
    # so the one which contains the most pixels
    bestCnt = contours[0]
    if not contours:
        raise ValueError("Could not detect a contour on the leaf")
    elif len(contours) > 1 :
        bestArea = 0
        for cont in contours:
            areaC = cv2.contourArea(cont)
            if areaC > bestArea:
                bestCnt = cont
        contour = bestCnt
    else:
        contour = contours[0]

    return contour


def get_leaf_perimeter(contour: MatLike) -> float:
    """
    The function retrives the length of the leaf perimeter
    ---------------------------------------------------------------------

    PARAMETERS
    ----------
    - contour: the OpenCV contour of the leaf

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The length of the perimeter, as a floating point number 
    """

    perimeter = cv2.arcLength(contour, False)

    return perimeter


def get_leaf_convexity(contour: MatLike) -> float:
    """
    The function computes the convexity of the leaf
    ---------------------------------------------------------------------

    PARAMETERS
    ----------
    - contour: the OpenCV contour of the leaf

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
