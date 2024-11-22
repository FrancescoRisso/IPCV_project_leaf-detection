import cv2
import numpy as np
from cv2.typing import MatLike
import math
from typing import Tuple


def get_top_tip_angle(thImg: MatLike) -> float:
    """
    Returns the the top tip angle of the leaf passed as a tresholded
    image. It uses the Hough tranform to find the segments that compose
    the edge of the leaf and elaborates the ones at the top to obtain
    the angle of the higher tip of the leaf.

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - thImg: the mask (bitmap) of the leaf

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The angle of the top tip, expressed in degrees
    """

    kernel = np.ones((3,3), np.uint8)
    # compute the leaf edge
    leafEdge = cv2.morphologyEx(thImg, cv2.MORPH_GRADIENT, kernel)

    
    lines = cv2.HoughLinesP(leafEdge, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=30)

    if len(lines) < 1:
        raise ValueError("The hough transform has not been able to locate any segment. Please check the imput")

    highLines = []
    NUM_HIGH_SEGMENTS = 14

    '''
    The idea now is to iterate over all the segments found by the houghLines functions
    and keep (by saving them in the ordered list highLines) only the 
    NUM_HIGH_SEGMENTS higher, so hte ones with the lowest y value
    '''

    # chose if a segment should be in the list highLines
    def __insertHigher(highLines, line):
        if len(highLines) < NUM_HIGH_SEGMENTS:
            highLines.append([line[0], line[1], line[2], line[3]])
            highLines.sort(key=lambda line: min(line[1], line[3]))
        else :
            # if the new segment is higher on the image than the lowest from the highLines list 
            # I insert it and sort the list, then pop the last one
            if min(line[1], line[3]) < min(highLines[-1][1], highLines[-1][3]):
                highLines.append([line[0], line[1], line[2], line[3]])
                highLines.sort(key=lambda line: min(line[1], line[3]))
                highLines.pop()

    # scan the list of segments found by hough transform  and keep
    # the top NUM_HIGH_SEG (higher in the image), so with the lowest y value 
    for line in lines:
        __insertHigher(highLines, line[0])

    # determines if a segment has a positive angular coefficient
    def __isIncr(seg):
        x1, y1, x2, y2 = seg
        if (x1<=x2 and y1<y2) or (x1>=x2 and y1>y2) :
            return 1
        return 0

    # determines if a segment has a negative angular coefficient
    def __isDecr(seg):
        x1, y1, x2, y2 = seg
        if (x1<=x2 and y1>y2) or (x2<=x1 and y2>y1):
            return 1
        return 0
    
    # determines if a segment is flat or almost flat (using the param)
    def __isFlat(seg):
        MIN_FLAT_ANGLE = 1.39626 #ca. 80 deg, is considered flat from 80 
        x1, y1, x2, y2 = seg
        if x1 == x2 :
            return 0
        elif (y1==y2) or (__getAngle(seg) > MIN_FLAT_ANGLE):
            return 1
        return 0

    # calculate the angle between the segment and the vertical line
    # that passes trough the segment top point
    def __getAngle(seg) :
        x1, y1, x2, y2 = seg
        if x1 == x2:
            return 0
        elif y1 == y2:
            return math.pi/2
        
        #calculate the top angle between the vertical and the segment
        return math.atan(abs(x2-x1)/abs(y2-y1))
    
    # check if two segment have the same angle 
    def __gotSameAngle(seg1, a1, seg2, a2):
        MIN_ANGLE = 0.2356  # ca. 14 deg, the minimum difference 2 segments should have to be considered different

        # we check for angle similarity only if the two segments have the angular coeff. of the same sign, 
        # it would be usless otherwise, the segments would certainly be different
        if (__isIncr(seg1) and __isIncr(seg2)) or (__isDecr(seg1) and __isDecr(seg2)):
            if (a1 == a2) or (abs(a2-a1) < MIN_ANGLE) :
                return 1
        elif a1+a2 < MIN_ANGLE:
            return 1
        return 0
    
    # check if the two segment can be used to identify an angle or if they are
    # two detections by hough of the same curved line
    def __isSameLine(seg1, seg2) :
        a1 = __getAngle(seg1)
        a2 = __getAngle(seg2)

        # we check if the 2 seg have more or less the same angle and if they have both vertex that are too close
        if (abs(seg1[0]-seg2[0])<8 and abs(seg1[2]-seg2[2])<8):
            return 1
        elif __gotSameAngle(seg1, a1, seg2, a2):
            return 1
        else:
            return 0
    
    '''
    In this for cycle we choose the two segments which will compose the angle, and then
    we compute its value
    '''
    # extract the top segment, it will be the first of the two segments which identify the top tip
    seg1 = highLines[0]
    highLines.pop(0)
    tipAngle = 0.0
    for seg2 in highLines:
        # we search for a second segment that we can use to identify and compute the angle of the top tip
        if not __isSameLine(seg1, seg2):
            # case1: the two segments have an angular coefficient with different sign (or one of them is flat),
            #       so we have to sum the angles of the two segments
            if (__isFlat(seg1) or __isFlat(seg2)) or (__isIncr(seg1) and __isDecr(seg2) ) or (__isDecr(seg1) and __isIncr(seg2) ):
                tipAngle = __getAngle(seg1) + __getAngle(seg2)
                break
            # case2: the two segments have an angular coeff of the same sign, we subtract the two angles found
            else:
                ang1 = __getAngle(seg1)
                ang2 = __getAngle(seg2)
                tipAngle = max(ang1, ang2) - min(ang1, ang2)
                break

    if not tipAngle:
        raise ValueError("Couldn't detect the tip of the leaf")
    
    # conversion to degrees
    tipAngle = tipAngle/math.pi*180

    return tipAngle
