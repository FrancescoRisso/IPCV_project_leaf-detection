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


#cnt = contours[4]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 5)
cv2.drawContours(img, contour, -1, (0,0,255), 8)

###
#
#   FEATURE EXTRACTION
#
###

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




plt.subplot(1,3,1)
plt.imshow(mask, cmap="grey")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edge, cmap="grey")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")


plt.show()
