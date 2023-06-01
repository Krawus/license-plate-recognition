import numpy as np
import cv2
from operator import itemgetter
import time

def calculateEstimatedPlateArea(imageWidth):
    return (0.3*imageWidth * 0.3*imageWidth / 4.7)

def getPlateContour(imageHeight, imageWidth, imageContours, estimatedMinPlateArea):
        #init the plate contour
    plateContour = None
    plateContourArea = imageHeight*imageWidth
    ratio = 0
    #iterate through contours
    for cnt in imageContours:

        hull = cv2.convexHull(cnt)
        #approx contour with quadrangle
        approx = cv2.approxPolyDP(hull,  0.04 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            if (cv2.contourArea(cnt) >= estimatedMinPlateArea):
                rect = cv2.minAreaRect(cnt)
                w = 0
                h = 0
                rectDim = rect[1]
                if rectDim[0] > rectDim[1]:
                    w = rectDim[0]
                    h = rectDim[1]
                else:
                    w = rectDim[1]
                    h = rectDim[0]
                
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if w/h >= ratio:
                    ratio = w/h        
                    if cv2.contourArea(cnt) < plateContourArea:
                        plateContour = hull

    return plateContour

def cutPlateROI(grayImg, plateContour, ROIoffset):
    # extreme points of the plate's contour
    leftmost = tuple(plateContour[plateContour[:,:,0].argmin()][0])[0] - ROIoffset
    rightmost = tuple(plateContour[plateContour[:,:,0].argmax()][0])[0] + ROIoffset
    topmost = tuple(plateContour[plateContour[:,:,1].argmin()][0])[1] - ROIoffset
    bottommost = tuple(plateContour[plateContour[:,:,1].argmax()][0])[1] + ROIoffset

    # CUT PLATE ROI 
    plateROI = grayImg[topmost:bottommost, leftmost:rightmost]

    return plateROI
    

def getPlateWhiteAreaHull(plateROIimg, plateContours, estimatedMinPlateArea):
    height, width = plateROIimg.shape
    plateHull = None
    plateHullArea = height*width
    #find plate white area hull
    for cnt in plateContours:
        if (cv2.contourArea(cnt) >= estimatedMinPlateArea): 

            hull = cv2.convexHull(cnt)
            # cv2.drawContours(plateGrayBGR,[plateHull], 0,(0,255,0),2)

            if (cv2.contourArea(hull) < plateHullArea):
                plateHull = hull

    # get hull points in list format
    plateHull = plateHull[:, 0, :]
    # get only hull points on left and right, skip the middle hull points 
    plateHull = list(filter(lambda hullPoint: (hullPoint[0] <= width/9) or (hullPoint[0] >= width-width/9), plateHull))
    return plateHull

def getImgCorners(imageGray):

    height, width = imageGray.shape
    return [[0,0], [width, 0], [width, height], [0, height]]


def getPlateWhiteAreaCorners(plateROIcorners, plateHull, plateROIGray):
    plateCorners = [[0,0], [0,0], [0,0], [0,0]]
    height, width = plateROIGray.shape
    for cornerIndex in range(len(plateROIcorners)):
        minDist = width
        xROI = plateROIcorners[cornerIndex][0]
        yROI = plateROIcorners[cornerIndex][1]
        for hullPoint in plateHull:
            xHull = hullPoint[0]
            yHull = hullPoint[1]

            #euclidean
            # distanceFromCorner = math.sqrt((xROI - xHull)**2 + (yROI - yHull)**2)
            #manhattan
            distanceFromCorner = abs(xROI - xHull) + abs(yROI - yHull)

            if (distanceFromCorner < minDist):
                minDist = distanceFromCorner
                plateCorners[cornerIndex] = [xHull, yHull]
    
    return plateCorners


def checkIfTwoLetter(letterBoxes):
        #distance between letterBox 0 and 1
    x0, y0, w0, h0 = letterBoxes[0]
    x1, y1, w1, h1 = letterBoxes[1]
    distBetweenLetters01 = x1 - (x0 + w0)

    x2, y2, w2, h2 = letterBoxes[2]
    distBetweenLetters12 = x2 - (x1 + w1)
    
    isTwoLetter = False
    if (distBetweenLetters12 >= distBetweenLetters01*2):
        isTwoLetter = True
    else:
        isTwoLetter = False

    return isTwoLetter


def getLettersBoundingBoxes(binaryWarpedPlateImg):
    lettersContours, hierarchy = cv2.findContours(binaryWarpedPlateImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plateHeight, plateWidth  = binaryWarpedPlateImg.shape
    letterBoxes = []
        
    for cnt in lettersContours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (h >= 0.69*plateHeight) and (w <= 0.3*plateWidth) and (w >= 0.05*plateWidth):
            letterBoxes.append([x,y,w,h])

    #sort letter boxes from left to right
    letterBoxes = sorted(letterBoxes, key=itemgetter(0))

    return letterBoxes

#dictionary with horizontal bounds of every letter on template image
LETTERS_DICT = {
    "A": (0, 35),
    "B": (35, 85),
    "C": (85, 135),
    "D": (135, 182),
    "E": (182, 231),
    "F": (231, 278),
    "G": (278, 327),
    "H": (327, 380),
    "I": (380, 423),
    "J": (423, 467),
    "K": (467, 518),
    "L": (518, 565),
    "M": (565, 614),
    "N": (614, 663),
    "O": (663, 710),
    "P": (710, 757),
    "R": (757, 807),
    "S": (807, 855),
    "T": (855, 902),
    "U": (902, 951),
    "V": (951, 998),
    "W": (998, 1046),
    "X": (1046, 1094),
    "Y": (1094, 1141),
    "Z": (1141, 1191),
    "0": (1191, 1235),
    "1": (1235, 1272),
    "2": (1272, 1317),
    "3": (1317, 1357),
    "4": (1357, 1401),
    "5": (1401, 1441),
    "6": (1441, 1484),
    "7": (1484, 1525),
    "8": (1525, 1567),
    "9": (1567, 1600)
}

def getLetterString(topLeft, bottomRight):

    detectedLetterBegin = topLeft[0]
    detectedLetterEnd = bottomRight[0]

    for letter, [leftBound, rightBound] in LETTERS_DICT.items():
        if(detectedLetterBegin >= leftBound and detectedLetterEnd <= rightBound):
            return letter

    return "0"


def swapFakeO(isTwoLetter, plateIdString):
    if isTwoLetter:
        for index in range(2):
            if plateIdString[index] == "0":
                plateIdString = plateIdString[:index] + "O" + plateIdString[index+1:]
                
        for index in range(2, len(plateIdString)):
            if plateIdString[index] == "O":
                plateIdString = plateIdString[:index] + "0" + plateIdString[index+1:]
    else:
        for index in range(3):            
            if plateIdString[index] == "0":
                plateIdString = plateIdString[:index] + "O" + plateIdString[index+1:]

        for index in range(3, len(plateIdString)):
            if plateIdString[index] == "O":
                plateIdString = plateIdString[:index] + "0" + plateIdString[index+1:]

    return plateIdString

def swapFake5(isTwoLetter, plateIdString):
    if isTwoLetter:
        for index in range(2):
            if plateIdString[index] == "5":
                plateIdString = plateIdString[:index] + "S" + plateIdString[index+1:]
    else:
        for index in range(3):            
            if plateIdString[index] == "5":
                plateIdString = plateIdString[:index] + "S" + plateIdString[index+1:]

    return plateIdString
    


def perform_processing(image: np.ndarray) -> str:

    try:
        imgOrg = cv2.resize(image, (0, 0), fx = 0.17, fy = 0.17)

        #convert img to gray
        imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)

        cannyValLow = 23
        cannyValHigh = 255
        gaussKernel = 7
        gaussParam = 1
        kernelDilSize = 2

        #gauss blur the image
        imgBlurred = cv2.GaussianBlur(imgGray, (gaussKernel,gaussKernel), gaussParam)
        imgCanny = cv2.Canny(imgBlurred, cannyValLow, cannyValHigh)

        #dilate image - to close contours
        kernelDil = np.ones((kernelDilSize,kernelDilSize),np.uint8)
        dilationImgCanny = cv2.dilate(imgCanny, kernelDil, iterations=1)

        #get the height and width of processed image to calculate area of contours to count as plate
        imgHeight, imgWidth = imgGray.shape

        #white area of plate -> 466x100mm -> ratio ~4.66
        estMinPlateArea = calculateEstimatedPlateArea(imgWidth)
        #create contours of img
        contoursImgCanny, hierarchy = cv2.findContours(dilationImgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #get plate contour
        plateContour = getPlateContour(imgHeight, imgWidth, contoursImgCanny, estMinPlateArea)

        #cut plate ROI
        ROIoffset = 15
        plateROI = cutPlateROI(imgGray, plateContour, ROIoffset)

        adaptiveKerPlate = 11
        subtractPlate = 2
        gaussKernelPlate = 5
        gaussParamPlate = 0
        kernelErSizePlate = 4

        # find potential contours of white area of the plate
        plateBlurred = cv2.GaussianBlur(plateROI, (gaussKernelPlate, gaussKernelPlate), gaussParamPlate)
        plateBinary = cv2.adaptiveThreshold(plateBlurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, adaptiveKerPlate, subtractPlate)

        kernelErodePlate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelErSizePlate,kernelErSizePlate))

        plateErosion = cv2.morphologyEx(plateBinary, cv2.MORPH_OPEN, kernelErodePlate)

        contoursPlate, hierarchy = cv2.findContours(plateErosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plateWhiteAreaHull = getPlateWhiteAreaHull(plateROI, contoursPlate, estMinPlateArea)

        #LU, RU, RB, LB
        plateROIcorners = getImgCorners(plateROI)
        plateWhiteAreaCorners = getPlateWhiteAreaCorners(plateROIcorners, plateWhiteAreaHull, plateROI)

         # white area dim -> 466x100mm
        # WARP THE PLATE
        plateROIheight, plateROIwidth = plateROI.shape
        warpCorners = np.float32([[0,0], [plateROIwidth, 0], [plateROIwidth, int(plateROIwidth/4.66)], [0, int(plateROIwidth/4.66)]])
        M = cv2.getPerspectiveTransform(np.float32(plateWhiteAreaCorners),warpCorners)
        warpedPlate = cv2.warpPerspective(plateROI,M,(plateROIwidth, int(plateROIwidth/4.66)))

        warpedPlateCannyUp = 255
        warpedPlateCannyDown = 131
        gaussKernelWarped = 7
        gaussParamWarped = 1

        #Threshold the warped plate
        warpedPlateBlurred = cv2.GaussianBlur(warpedPlate, (gaussKernelWarped,gaussKernelWarped), gaussParamWarped)
        cannyWarpedPlate =  cv2.Canny(warpedPlateBlurred,warpedPlateCannyDown,warpedPlateCannyUp)
        kernelMorphLetters = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        cannyWarpedPlate = cv2.morphologyEx(cannyWarpedPlate, cv2.MORPH_DILATE, kernelMorphLetters)

        letterBoxes = getLettersBoundingBoxes(cannyWarpedPlate)

        #check type of plate  -> part of the place discriminator could be 2 or 3 letters long
        isTwoLetter = checkIfTwoLetter(letterBoxes)

        allLettersImg = cv2.imread('processing/lettersTemplate.png', 0)
        ret, allLettersImg = cv2.threshold(allLettersImg, 127, 255,0)

        # blur the warped plate
        warpedPlateBlurred = cv2.GaussianBlur(warpedPlate, (3,3), 0)

        #kernel to morph open every letter
        kernelLetter = np.ones((5,5),np.uint8)

        plateId = ""
        for box in letterBoxes:
            #cut letter based on bounding box
            x,y,w,h = box
            letter = warpedPlateBlurred[y:y+h, x:x+w]
            ret, letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            letter = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernelLetter)

            #get the cut letter ratio
            ratioCutLetter = w/h
            #40 is height of the template letter
            letter = cv2.resize(letter, (int(ratioCutLetter*40), 40))

            w, h = letter.shape[::-1]
            res = cv2.matchTemplate(allLettersImg,letter,cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            plateId += getLetterString(top_left, bottom_right)

        # number 0 could not be in the part of the place discriminator, letter O coult not be in the right part of plate
        plateId = swapFakeO(isTwoLetter, plateId)
        # number 5 could not be in the part of the place discriminator -> minimizing the likelihood of confusion between 5 and S
        plateId = swapFake5(isTwoLetter, plateId)

    except:
        plateId = "PO12345"


    return plateId
