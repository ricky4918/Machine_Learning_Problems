import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########Variables###########
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
vol = 0
volBar =400
volPer = 0
detector = htm.handDetector(detectionCon=0.7)
# Volume interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate( IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange() #(-63.5, 0)
minVol = volRange[0]
maxVol = volRange[1]
##############################



while True:
    ########################################################
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=False)
    ########################################################

    if len(lmlist) != 0:

        x1, y1 = lmlist[4][1], lmlist[4][2] #thumb finger
        x2, y2 = lmlist[8][1], lmlist[8][2] #index finger
        cx,cy = (x1+x2)//2, (y1+y2)//2 # center point

        # put circle on index, thumb
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED )
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # line between thumb and center
        cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3)
        # put circle on center
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        # calculate the distance bettwen thumb and index finger
        length = math.hypot(x2-x1, y2-y1)
        cv2.putText(img, f'Length: {int(length)}', (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 125, 0), 3)

        # make center circle green when two fingers are closed
        if length <50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)


        # Hand Range: (12,150)
        # Volume Range (-65, 0)
        #convert length in Volume range
        vol = np.interp(length, [12,150], [minVol, maxVol])
        #convert length in bar range
        volBar = np.interp(length, [12, 150], [400, 150])
        #convert length in percentage range
        volPer = np.interp(length, [12, 150], [0, 100])

        #set volume
        volume.SetMasterVolumeLevel(vol, None)

    #displays for bar and percentage
    cv2.rectangle(img, (20, 150), (85,400), (0,255,0), 3)
    cv2.rectangle(img, (20, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Volume: {int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3)


    #display Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,125,0), 3)

    #display
    cv2.imshow("Img", img)
    cv2.waitKey(1)