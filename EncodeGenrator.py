import cv2
import os
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://realtimefacedetection-42014-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "realtimefacedetection-42014.appspot.com"
})


folderImagesPath = 'Resoureces/Images'
storagePath = 'Images'
imagesPathList = os.listdir(folderImagesPath)

# print(imagesPathList)

imgList = []
idList = []

for path in imagesPathList:
    imgList.append(cv2.imread(os.path.join(folderImagesPath, path)))
    idList.append(os.path.splitext(path)[0])

    fileName = f'{folderImagesPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # print(path)
    # print(os.path.splitext(path)[0])

print(idList)


def findEncodings(ImagesList):
    EncodingList = []
    for img in ImagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        EncodingList.append(encode)

    return EncodingList


print("Encoding statrted")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithId = [encodeListKnown, idList]
# print(encodeListKnown)
print("Encoding Completed")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithId, file)
file.close()

print("EncodeingFile Saved")
