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
'''
plt.subplot(1, 2, 1)
plt.imshow(thImg, cmap="grey")
plt.axis("off")
plt.title("th")'''

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

# !!! da gestire errore per accesso fuori dall'immagine

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


img = cv2.line(img, (marginL, imgH), (marginL, 0), (0,255, 0), 10)
img = cv2.line(img, (marginR, imgH), (marginR, 0), (0,255, 0), 10)
img = cv2.line(img, (0, marginT), (imgW, marginT), (0,255, 0), 10)
img = cv2.line(img, (0, marginB), (imgW, marginB), (0,255, 0), 10)

print("MarginL: ", marginL, " - MarginR: ", marginR, " - MarginT: ", marginT, " - MarginB: ", marginB)

##
##  Classificate the lines found via hough by linking them to the side of the sheet they belong
##

#PERCENTUALE_DISTANZA_LATO = 1
DIST_PERC = 1

# the values i will use to extract the roi in such a way that 
# only the sheet of paper and the leaf are in
roiL = 0; roiR = imgW; roiT = 0; roiB = imgH

def isVertical(points):
    if (line[2]-line[0]) < 5 : 
        return 1 
    return -1

def isHorizontal(points):
    if (line[2]-line[0]) < 5 : 
        return 1 
    return 0

def belongsToSide(point, side) :
    if (point < side + int(((DIST_PERC/100)*imgW)) ) and ( point > side - int(((DIST_PERC/100)*imgW)) ) :
        return 1
    return 0

for line in lines:
    line = line[0]
    if isVertical(line) :

        # check if the line belongs to the left side
        if belongsToSide(line[0], marginL) or belongsToSide(line[2], marginL) :
            # if true I eventually update roiL with a more conservative value
            if max(line[0], line[2]) > roiL :
                roiL = max(line[0], line[2])
        
        elif belongsToSide(line[0], marginR) or belongsToSide(line[2], marginR) :
            if min(line[0], line[2]) < roiR :
                roiR = min(line[0], line[2])
    
    elif isHorizontal(line) :

        if belongsToSide(line[1], marginT) or belongsToSide(line[3], marginT) :
            if max(line[1], line[3]) > roiT :
                roiT = max(line[1], line[3])
        
        elif belongsToSide(line[1], marginB) or belongsToSide(line[3], marginB) :
            if min(line[1], line[3]) < roiB :
                roiB = min(line[1], line[3])

plt.subplot(1, 2, 1)
plt.imshow(img[roiT:roiB , roiL:roiR])
plt.axis("off")
plt.title("exact roi")

print("roiL: ", roiL, " - roiR: ", roiR, " - roiT: ", roiT, " - roiB: ", roiB)


plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("bordo")


plt.show()