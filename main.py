import os
import pickle
import cvzone
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://realtimefacedetection-42014-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "realtimefacedetection-42014.appspot.com"
})




cap = cv2.VideoCapture(0)
# camp = cv2.flip(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Background Images
imgBack = cv2.imread('Resoureces/BackGround/frame.jpg')

# All Images with differents modes

folderModePath = 'Resoureces/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []


for path in modePathList:
    imgMode = cv2.imread(os.path.join(folderModePath,path))
    imgMode = cv2.resize(imgMode, (250,200))
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

    imgBack[50:50+480 , 45:45+640] = img
    imgBack[100:100+200, 900:900+250] = imgModeList[0]

    for encodeFace, faceLoc in zip(encodefaceCurFrame,faceCurFrame):
        match = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        # print("match", match)
        # print("Dis", faceDis)

        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            print("Matched Succesfully")
            print(idList[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            bbox = 50 + x1, 45 + y1, x2-x1, y2-y1

            imgBack = cvzone.cornerRect(imgBack, bbox, rt=0)

        else: print("Not Matched")





    # cv2.imshow("WebCam", img)
    cv2.imshow("Face Recoginition", imgBack)
    cv2.waitKey(1)