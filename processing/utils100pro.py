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
    #iterate through contours
    for cnt in imageContours:
        #approx contour with quadrangle
        approx = cv2.approxPolyDP(cnt,  0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            if (cv2.contourArea(approx) >= estimatedMinPlateArea):
                if cv2.contourArea(cnt) < plateContourArea:
                    plateContour = cnt

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


def getPlateWhiteAreaCorners(plateROIcorners, plateHull):
    plateCorners = [[0,0], [0,0], [0,0], [0,0]]
    width = plateROIcorners[1][0]
    
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
        if (h >= 0.7*plateHeight) and (w <= 0.3*plateWidth):
            # cv2.rectangle(imgToDrawLetters,(x,y),(x+w,y+h),(0,255,0),2)
            letterBoxes.append([x,y,w,h])

    #sort letter boxes from left to right
    letterBoxes = sorted(letterBoxes, key=itemgetter(0))

    return letterBoxes


def getLetterString(topLeft, bottomRight):

    letterBegin = topLeft[0]
    letterEnd = bottomRight[0]

    if (letterBegin >= 0 and letterEnd <= 35):
        return "A"
    elif (letterBegin >= 35 and letterEnd <= 85):
        return "B"
    elif (letterBegin >= 85 and letterEnd <= 135):
        return "C"
    elif (letterBegin >= 135 and letterEnd <= 182):
        return "D"
    elif (letterBegin >= 182 and letterEnd <= 231):
        return "E"
    elif (letterBegin >= 231 and letterEnd <= 278):
        return "F"
    elif (letterBegin >= 278 and letterEnd <= 327):
        return "G"
    elif (letterBegin >= 327 and letterEnd <= 380):
        return "H"
    elif (letterBegin >= 380 and letterEnd <= 423):
        return "I"
    elif (letterBegin >= 423 and letterEnd <= 467):
        return "J"
    elif (letterBegin >= 467 and letterEnd <= 518):
        return "K"
    elif (letterBegin >= 518 and letterEnd <= 565):
        return "L"
    elif (letterBegin >= 565 and letterEnd <= 614):
        return "M"
    elif (letterBegin >= 614 and letterEnd <= 663):
        return "N"
    elif (letterBegin >= 663 and letterEnd <= 710):
        return "O"
    elif (letterBegin >= 710 and letterEnd <= 757):
        return "P"
    elif (letterBegin >= 757 and letterEnd <= 807):
        return "R"
    elif (letterBegin >= 807 and letterEnd <= 855):
        return "S"
    elif (letterBegin >= 855 and letterEnd <= 902):
        return "T"
    elif (letterBegin >= 902 and letterEnd <= 951):
        return "U"
    elif (letterBegin >= 951 and letterEnd <= 998):
        return "V"    
    elif (letterBegin >= 998 and letterEnd <= 1046):
        return "W" 
    elif (letterBegin >= 1046 and letterEnd <= 1094):
        return "X" 
    elif (letterBegin >= 1094 and letterEnd <= 1141):
        return "Y" 
    elif (letterBegin >= 1141 and letterEnd <= 1191):
        return "Z" 
    elif (letterBegin >= 1191 and letterEnd <= 1235):
        return "0" 
    elif (letterBegin >= 1235 and letterEnd <= 1272):
        return "1" 
    elif (letterBegin >= 1272 and letterEnd <= 1317):
        return "2" 
    elif (letterBegin >= 1317 and letterEnd <= 1357):
        return "3" 
    elif (letterBegin >= 1357 and letterEnd <= 1401):
        return "4" 
    elif (letterBegin >= 1401 and letterEnd <= 1441):
        return "5" 
    elif (letterBegin >= 1441 and letterEnd <= 1484):
        return "6" 
    elif (letterBegin >= 1484 and letterEnd <= 1525):
        return "7" 
    elif (letterBegin >= 1525 and letterEnd <= 1567):
        return "8" 
    elif (letterBegin >= 1567 and letterEnd <= 1600):
        return "9" 


def swapFakeO(isTwoLetter, plateIdString):
    if isTwoLetter:
        print("twoletter")
        
        for index in range(2):
            if plateIdString[index] == "0":
                plateIdString = plateIdString[:index] + "O" + plateIdString[index+1:]
                
        for index in range(2, len(plateIdString)):
            if plateIdString[index] == "O":
                plateIdString = plateIdString[:index] + "0" + plateIdString[index+1:]
    else:
        print("threeletter")
        for index in range(3):            
            if plateIdString[index] == "0":
                plateIdString = plateIdString[:index] + "O" + plateIdString[index+1:]

        for index in range(3, len(plateIdString)):
            if plateIdString[index] == "O":
                plateIdString = plateIdString[:index] + "0" + plateIdString[index+1:]

    return plateIdString




def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    print("------------------")
    try:

        start = time.time()
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

        print("CONTOURS FOUND")
        #get plate contour

        plateContour = getPlateContour(imgHeight, imgWidth, contoursImgCanny, estMinPlateArea)

        print("plate contour FOUND")

        imgOrgCpy = imgOrg.copy()

        cv2.drawContours(imgOrgCpy, [plateContour], 0, (0, 255, 0), 2)
        cv2.imshow("whole plate contour", imgOrgCpy)

        

        #cut plate ROI
        ROIoffset = 15
        plateROI = cutPlateROI(imgGray, plateContour, ROIoffset)
        cv2.imshow("ROI CUT", plateROI)

        adaptiveKerPlate = 11
        subtractPlate = 2
        gaussKernelPlate = 5
        gaussParamPlate = 0
        kernelErSizePlate = 3

        # find potential contours of white area of the plate
        plateBlurred = cv2.GaussianBlur(plateROI, (gaussKernelPlate, gaussKernelPlate), gaussParamPlate)
        plateBinary = cv2.adaptiveThreshold(plateBlurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, adaptiveKerPlate, subtractPlate)
    
        kernelErodePlate = np.ones((kernelErSizePlate,kernelErSizePlate),np.uint8)
        plateErosion = cv2.erode(plateBinary, kernelErodePlate)

        contoursPlate, hierarchy = cv2.findContours(plateErosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        plateHull = getPlateWhiteAreaHull(plateROI, contoursPlate, estMinPlateArea)
        print("len(plateHull)" ,len(plateHull))
        #LU, RU, RB, LB
        plateROIcorners = getImgCorners(plateROI)

        plateCorners = getPlateWhiteAreaCorners(plateROIcorners, plateHull)
    #466x100mm

        # WARP THE PLATE
        height, width = plateROI.shape
        warpCorners = np.float32([[0,0], [width, 0], [width, int(width/4.66)], [0, int(width/4.66)]])
        M = cv2.getPerspectiveTransform(np.float32(plateCorners),warpCorners)
        warpedPlate = cv2.warpPerspective(plateROI,M,(width, int(width/4.66)))

        adaptiveKerWarpedPlate = 11
        subtractWarpedPlate = 2
        gaussKernelWarped = 5
        gaussParamWarped = 0

        #Threshold the warped plate
        warpedPlateBlurred = cv2.GaussianBlur(warpedPlate, (gaussKernelWarped,gaussKernelWarped), gaussParamWarped)
        ret, binaryWarpedPlate = cv2.threshold(warpedPlateBlurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernelMorphLetters = np.ones((2,2),np.uint8)
        binaryWarpedPlate = cv2.morphologyEx(binaryWarpedPlate, cv2.MORPH_OPEN, kernelMorphLetters)


        cv2.imshow("warpedBinary", binaryWarpedPlate)

        lettersContours, hierarchy = cv2.findContours(binaryWarpedPlate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plateHeight, plateWidth  = binaryWarpedPlate.shape

        letterBoxes = getLettersBoundingBoxes(binaryWarpedPlate)

        #check type of plate
        isTwoLetter = checkIfTwoLetter(letterBoxes)

                
        allLettersImg = cv2.imread('processing/lettersPlate.png', 0)
        ret, allLettersImg = cv2.threshold(allLettersImg, 127, 255,0)

        kernelLetter = np.ones((1,1),np.uint8)

        plateId = ""
        counter = 0
        for box in letterBoxes:
            #cut letter based on bounding box
            x,y,w,h = box
            letter = cv2.bitwise_not(binaryWarpedPlate[y:y+h, x:x+w])
            letter = cv2.erode(letter, kernelLetter)
            #get the cut letter ratio
            ratioCutLetter = w/h
            #40 is height of the template letter
            letter = cv2.resize(letter, (int(ratioCutLetter*40), 40))
            cv2.imshow(f"{counter}", letter)

            w, h = letter.shape[::-1]
            res = cv2.matchTemplate(allLettersImg,letter,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            plateId += getLetterString(top_left, bottom_right)
            counter+=1

        plateId = swapFakeO(isTwoLetter, plateId)

        end = time.time()

        print("processing time: ", end-start)
        print(plateId)
    except:
        print("failed to processImage!!")
        plateId = "PO12345"

    key = ord(" ")
    while key != ord("d"):
        key = cv2.waitKey(5)
    

    return plateId
