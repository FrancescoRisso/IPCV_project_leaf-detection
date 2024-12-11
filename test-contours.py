import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from typing import Tuple
from functions.utils.leaf import get_leaf_mask
from functions.lengths.paper_roi import find_roi_boundaries

base_path  = "C:/Users/Utente/Desktop/scuola/universita/ImgProcessing/IPCV-leaf.detection/IPCV_project_leaf-detection/dataset/images"
foglia = "frassino/001.jpg"
img = cv2.imread(os.path.join(base_path, foglia))
l, r, t, b = find_roi_boundaries(img)
img = img[t:b, l:r]
mask = get_leaf_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

def find_leaf_contour(mask: MatLike) -> np.ndarray:
    """
    The function retrives the leaf contours using openCV findContours
    function, after applying a series of morphological operation to 
    find the right contour

    ---------------------------------------------------------------------
    PARAMETERS
    ----------

    - leafMask: the tresholded image of the leaf, a bitmap

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The main contour of the leaf, as a numpy array.
    It is accepted by the openCV functions which work with contours 
    """

    # a huge closing takes place, using a circular kernel. It is a necessary
    # passage, all the holes in the leaves are filled to avoid problems
    # like distinguish between different contours
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

    # N.B. the new coordinates are not usable on th eleaf maske as the image is bigger,
    #       however the features which will be extracted are not position related so we'll ignore that 

    edge = cv2.Canny(mask, 175, 175)
    edge = cv2.dilate(edge, ker2)   
    edge = cv2.erode(edge, ker2)
    edge = cv2.dilate(edge, ker3)  

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours is a list of all the detected contours

    # select the right contour by finding the biggest one, the one which contains the most pixels
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

###
#
#   FEATURE EXTRACTION
#
###
contour = find_leaf_contour(mask)

perimeter = cv2.arcLength(contour, False)

#isConvex = cv2.isContourConvex(contours)

perimeterTxt = f"Value: {perimeter:.1f}"

strOut = "pr: " + perimeterTxt
print(strOut, end=" ")
cv2.putText(img, strOut, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 8, (0,0,255), 18)

##  CONVEXITY

hull = cv2.convexHull(contour)

cv2.drawContours(img, [hull], -1, (255,0,0), 9)

areaContour = cv2.contourArea(contour)
areaHull = cv2.contourArea(hull)
print(" - hull: ", areaHull, " - cntArea: ",areaContour, " - diff: ",areaHull-areaContour)




plt.subplot(1,2,1)
plt.imshow(mask, cmap="grey")
plt.axis("off")


plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")


plt.show()

def get_leaf_contours(img: MatLike) -> Tuple[float] :
    """
    The function retrives the leaf contours using openCV findContours
    function, after applying a series of morphological operation to 
    find the right contour

    ---------------------------------------------------------------------
    PARAMETERS
    ----------

    - leafMask: the tresholded image of the leaf, a bitmap

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A tuple, composed of:
    - The main contour of the leaf, usable by other openCV functions
    """
    ker1 = np.ones((56, 56), np.uint8)
    ker2 = np.ones((6,6), np.uint8)
    ker3 = np.ones((2,2), np.uint8)
    kerR  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(104, 104))

    mask = cv2.dilate(mask, kerR)
    mask = cv2.erode(mask, kerR)

    # noise cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((12,12)))

    # add a 10px black border around the image, so the leaves that exceed the image
    # dimensions are properly elaborated by Canny and then by the findContour function
    mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

            #
            # WARNING: POSSIBLE COORDINATES PROBLEM WITH THAT APPROACH
            #

    edge = cv2.Canny(mask, 175, 175)
    edge = cv2.dilate(edge, ker2)   
    edge = cv2.erode(edge, ker2)
    edge = cv2.dilate(edge, ker3)  

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours is a list of all the detected contours, where each one of them is an array of point representing it
    # hierarchy

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



    return