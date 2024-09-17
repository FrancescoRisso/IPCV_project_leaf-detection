from functions.utils.image import Image as im
from functions.utils.image import Formats
from functions.utils.segment import Segment
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = im.from_file("./IMG20240806153249.jpg")
imgG = img.as_monochromatic
imgG = cv2.blur(imgG,(8,8))
'''
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="grey")
plt.axis("off")
plt.title("og")'''

ret,thImg = im.threshold(imgG,130,255)

ker1 = np.ones((26,26), np.uint8)

thImg = im.erode(thImg, ker1)
thImg = im.dilate(thImg, ker1)
'''
plt.subplot(2, 2, 2)
plt.imshow(thImg, cmap="grey")
plt.axis("off")
plt.title("th")'''

ker = np.ones((3,3), np.uint8)
#smaller = cv2.erode(thImg, mask)
#cv2.bitwise_and(thImg, smaller, contour)

contour = cv2.morphologyEx(thImg, cv2.MORPH_GRADIENT, ker)

#canny way
#contour = cv2.Canny(img,200,400)

contour = im.dilate(contour, ker)

plt.subplot(1, 2, 1)
plt.imshow(contour, cmap="grey")
plt.axis("off")
plt.title("th")

lines = cv2.HoughLinesP(contour , 1, np.pi/2, 50, minLineLength=150, maxLineGap=80)


print(lines)

for line in lines:
    line = line[0]
    img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,255), 15)

plt.subplot(1, 2, 2)
plt.imshow(img.as_RGB)
plt.axis("off")
plt.title("bordo")






plt.show()