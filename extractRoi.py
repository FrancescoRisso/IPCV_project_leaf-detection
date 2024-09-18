from functions.utils.segment import Segment
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./dataset/salvia/001.jpg")
imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgG = cv2.blur(imgG,(8,8))

'''
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="grey")
plt.axis("off")
plt.title("og")'''

ret,thImg = cv2.threshold(imgG,120,255, cv2.THRESH_BINARY)

ker1 = np.ones((26,26), np.uint8)

thImg = cv2.erode(thImg, ker1)
thImg = cv2.dilate(thImg, ker1)
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

contour = cv2.dilate(contour, ker)

plt.subplot(1, 2, 1)
plt.imshow(thImg, cmap="grey")
plt.axis("off")
plt.title("th")

lines = cv2.HoughLinesP(contour , 1, np.pi/2, 50, minLineLength=150, maxLineGap=80)

print(lines)
print("Detected lines: ", len(lines))
print("img shape: ",img.shape[0],img.shape[1])

for line in lines:
    line = line[0]
    img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,255), 15)

'''
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("bordo")'''

#
#   find the possible coortinates of the 4 side, margin Left, Right, Top, Bottom, given the tresholded image thImg
#

NUM_OF_SAMPLES = 30

imgH, imgW = img.shape[:2]
marginL = 0
marginR = imgW
marginT = 0
marginB = imgH


# trovo il margine sinistro: partendo dal bordo immagine avanzo fino al foglio per più (numOfSample) volte
# la coordinata x del margine sarà la mediana dei valori deltaX , cioè la mediana delle coordinate dei punti del bordo

samples = []
deltaY = 0
for i in range(NUM_OF_SAMPLES):

    deltaX = 0
    while thImg[imgH//2 + deltaY, 0 + deltaX] == 0 :    # N.B. !!! img(y, x)
        deltaX += 1
    
    samples.append(deltaX)

    deltaY += 8 #scendo a step di 8

samples.sort()
marginL = samples[len(samples)//2]
samples.clear()

img = cv2.line(img, (marginL, imgH), (marginL, 0), (0,255, 0), 15)

#
# margine destro
#

samples = []
deltaY = 0
for i in range(NUM_OF_SAMPLES):

    deltaX = imgW-1
    while thImg[imgH//2 + deltaY, 0 + deltaX] == 0 :    # N.B. !!! img(y, x)
        deltaX -= 1
    
    samples.append(deltaX)

    deltaY += 8 #scendo a step di 8

samples.sort()
marginR = samples[len(samples)//2]
samples.clear()

img = cv2.line(img, (marginR, imgH), (marginR, 0), (0,255, 0), 15)

#
# Margine Top
#

samples = []
deltaX = 0
for i in range(NUM_OF_SAMPLES):

    deltaY = 0
    while thImg[0 + deltaY, imgW//2 + deltaX] == 0 :    # N.B. !!! img(y, x)
        deltaY += 1
    
    samples.append(deltaY)

    deltaX += 6 #avanzo a step di 6

samples.sort()
marginT = samples[len(samples)//2]
samples.clear()

img = cv2.line(img, (0, marginT), (imgW, marginT), (0,255, 0), 15)

#
#   Margine Bottom
#

samples = []
deltaX = 0
for i in range(NUM_OF_SAMPLES):

    deltaY = imgH-1
    while thImg[0 + deltaY, imgW//2 + deltaX] == 0 :    # N.B. !!! img(y, x)
        deltaY -= 1
    
    samples.append(deltaY)

    deltaX += 6 #avanzo a step di 6

samples.sort()
marginB = samples[len(samples)//2]
samples.clear()

img = cv2.line(img, (0, marginB), (imgW, marginB), (0,255, 0), 15)

print("MarginL: ", marginL, " - MarginR: ", marginR, " - MarginT: ", marginT, " - MarginB: ", marginB)


plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("bordo")


plt.show()