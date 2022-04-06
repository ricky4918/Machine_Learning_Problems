import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

############ Variables ###################
pTime = 0;
cTime = 0;
cap = cv2.VideoCapture(0)
detector = htm.handDetector();
##########################################




while True:
    success, img = cap.read()

    #1. Find Hands' landmarks
    img = detector.findHands(img, draw=True)

    #2. Find Hands' landmarks' position
    lmList = detector.findPosition(img, draw=False)

    #3. Display Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    #4. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)