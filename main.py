import os
import pickle
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://realtimefacedetection-42014-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "realtimefacedetection-42014.appspot.com"
})

bucket = storage.bucket()


cap = cv2.VideoCapture(0)
# camp = cv2.flip(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Background Images
imgBack = cv2.imread('Resoureces/BackGround/frame.jpg')

# All Images with differents modes

folderModePath = 'Resoureces/Status'
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

modeType = 0
counter = 0
sId = -1
idImg = []

while True:
    success, img = cap.read()

    smallImg = cv2.resize(img,(0,0),None,0.25,0.25)
    smallImg = cv2.cvtColor(smallImg,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(smallImg)
    encodefaceCurFrame = face_recognition.face_encodings(smallImg,faceCurFrame)

    imgBack[30:30+480, 30:30+640] = img
    imgBack[170:170+200, 730:730+250] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodefaceCurFrame, faceCurFrame):
            match = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            # print("match", match)
            # print("Dis", faceDis)

            matchIndex = np.argmin(faceDis)

            if match[matchIndex]:
                print("Matched Succesfully")
                # print(idList[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 50 + x1, 45 + y1, x2 - x1, y2 - y1
                imgBack = cvzone.cornerRect(imgBack, bbox, rt=0)

                sId = idList[matchIndex]
                # print(sId)

                if counter == 0:
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                getIdInfo = db.reference(f'students/{sId}').get()

                ref = db.reference(f'students/{sId}')

                last_attendance_time_str = getIdInfo['last_attendance_time']
                datetimeObject = datetime.strptime(last_attendance_time_str,"%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                if secondsElapsed > 30:
                    ref = db.reference(f'students/{sId}')
                    getIdInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(getIdInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0

            if 10 < counter < 20:
                modeType = 2

            if counter <= 10:
                cv2.putText(imgBack, str(getIdInfo['name']), (720, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.putText(imgBack, str(getIdInfo['major']), (720, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

            counter += 1

            if counter >= 20:
                counter = 0
                modeType = 0
                getIdInfo = []
                idImg = []

    else:
        modeType = 0
        counter = 0
        print("Not Matched")

    # cv2.imshow("WebCam", img)
    cv2.imshow("Face Recoginition", imgBack)
    cv2.waitKey(1)