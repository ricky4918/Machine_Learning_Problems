import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm



############################
cap = cv2.VideoCapture(0)
brushTickness = 15
eraserTickness = 50
drawColor = (255,0,255)
wCam, hCam = 1280, 720
pTime =0
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector(detectionCon=0.4)
xp, yp = 0,0
##############################

# import image file
folderPath = "header"
mylist = os.listdir(folderPath)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]

# Create new Canvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2. Find Hand Landmarks

    img = detector.findHands(img, draw = False)
    lmList, bbox = detector.findPosition(img, draw = False)

    # tip of index and middle finger
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

    #3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

    #4. If selection mode - Two Fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            print("selection Mode")
            # Checking for the click
            if y1 < 125:
                    if 100<x1<200:
                        header = overlayList[0]
                        drawColor = (255,0,255)
                    elif 250<x1<450:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 550<x1<700:
                        header = overlayList[2]
                        drawColor = (0, 0, 255)
                    elif 750<x1<950:
                        header = overlayList[3]
                        drawColor = (0, 125, 0)
                    elif 1000<x1<1250:
                        header = overlayList[4]
                        drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

    #5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp== 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserTickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserTickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushTickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushTickness)
            xp,yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # # Display Frame Rate
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 125, 0), 3)


    # setting the header image
    img[0:125, 0:1280] = header

    #diaplay
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas,", imgCanvas)
    cv2.waitKey(1)