import os
import pickle
import cvzone
import numpy as np
import cv2
import face_recognition


cap = cv2.VideoCapture(0)
# camp = cv2.flip(1)
cap.set(3, 740)
cap.set(4, 480)

# Background Images
imgBack = cv2.imread('Resoureces/BackGround/back.jpg')

# All Images with differents modes

folderModePath = 'Resoureces/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []


for path in modePathList:
    imgMode = cv2.imread(os.path.join(folderModePath,path))
    imgMode = cv2.resize(imgMode, (300,200))
    imgModeList.append(imgMode)
    # imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))




print("Loading Encode File")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithId = pickle.load(file)
file.close()
encodeListKnown,idList = encodeListKnownWithId

# print(idList)

print("Encode File Loaded")

while True:
    success, img = cap.read()

    smallImg = cv2.resize(img,(0,0),None,0.25,0.25)
    smallImg = cv2.cvtColor(smallImg,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(smallImg)
    encodefaceCurFrame = face_recognition.face_encodings(smallImg,faceCurFrame)

    imgBack[200:200+480 , 80:80+640] = img
    imgBack[100:100+200, 50:50+300] = imgModeList[1]

    for encodeFace, faceLoc in zip(encodefaceCurFrame,faceCurFrame):
        match = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        # print("match", match)
        # print("Dis", faceDis)

        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            print("Matched Succesfully")
            print(idList[matchIndex])
            X1, Y1, X2, Y2 = faceLoc
            X1, Y1, X2, Y2 = X1*4, Y1*4, X2*4, Y2*4

            bbox = 80 + X1, 200 + Y1, X2-X1, Y2-Y1

            imgBack = cvzone.cornerRect(imgBack, bbox, rt=0)

        else: print("Not Matched")



    cv2.imshow("WebCam", img)
    cv2.imshow("Face Recoginition", imgBack)
    cv2.waitKey(1)
