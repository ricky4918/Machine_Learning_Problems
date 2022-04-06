import cv2
import time
import os
import HandTrackingModule as htm

############ Variables ################
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
pTime =0
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
########################################

# import image file
folderPath = "fingerimg"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # Find Hand Landmarks
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] <lmList[tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        h,w,c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20,255), (170,425), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    #display Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (460, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 125, 0), 3)

    #display
    cv2.imshow("Image", img)
    cv2.waitKey(1)