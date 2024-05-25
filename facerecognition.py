import math
from ultralytics import YOLO
import asyncio
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, db
import cvzone
import face_recognition
import time
model=YOLO('fakevsrealface.pt')
# إعداد Firebase
cred = credentials.Certificate("here your json file ")
firebase_admin.initialize_app(cred, {
    'databaseURL': "URL from databese",
    'storageBucket': 'from databese storage'
})
bucket = storage.bucket()
class_name=['device', 'live', 'mask', 'photo']
# إعدادات الكاميرا والمتحولات
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imgbackground = cv2.imread('Resources/background.jpg')
imgbackground2 = cv2.imread('Resources/background2.png')

# تحميل صور الوضعيات
floderModesPath = "Resources/Modes"
modePathList = os.listdir(floderModesPath)
imgModeList = [cv2.imread(os.path.join(floderModesPath, path)) for path in modePathList]

# تحميل بيانات الترميز
with open('Encoding.p', "rb") as file:
    encodelistknewwithids = pickle.load(file)
encodelistknew, studentIds = encodelistknewwithids

modetype = 0
counter = 0
id = -1
imgstud = []

async def fetch_student_info(student_id):
    # جلب بيانات الطالب من Firebase
    student_info = db.reference(f'Students/{student_id}').get()
    return student_info

async def fetch_student_image(student_id):
    # جلب صورة الطالب من Firebase
    blob = bucket.get_blob(f'Images/{student_id}.jpg')
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    img_student = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    return img_student

async def main():
    global counter, modetype, id, result_imgstud
    while True:
        ses, img = cap.read()
        res=model(img)
        for r in res:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
                w,h=x2-x1,y2-y1
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                print(class_name[cls])
                if class_name[cls]=='live' and conf>0.3:

                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                    faceCurframe = face_recognition.face_locations(imgS)
                    encoCurframe = face_recognition.face_encodings(imgS, faceCurframe)

                    imgbackground[162:162 + 480, 55:55 + 640] = img
                    imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
                    if faceCurframe:
                        for encodeface, faceloc in zip(encoCurframe, faceCurframe):
                            matches = face_recognition.compare_faces(encodelistknew, encodeface)
                            face_distance = face_recognition.face_distance(encodelistknew, encodeface)
                            match_index = np.argmin(face_distance)

                            if matches[match_index]:
                                x1, y1, x2, y2 = faceloc
                                x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
                                id = studentIds[match_index]

                                if counter == 0:
                                    cvzone.putTextRect(imgbackground, 'Loading', (275, 400))
                                    cv2.imshow('background', imgbackground)
                                    cv2.waitKey(1)
                                    counter = 1
                                    modetype = 1

                        if counter != 0:
                            if counter == 1:
                                task1 = asyncio.create_task(fetch_student_info(id))
                                task2 = asyncio.create_task(fetch_student_image(id))

                                # استخدام await لجلب البيانات غير المتزامنة
                                result_student_info = await task1
                                result_imgstud = await task2

                                # تحديث البيانات
                                datetime_object = datetime.strptime(result_student_info['last_attendance'],
                                                                    '%Y-%m-%d %H:%M:%S')
                                total_seconds = (datetime.now() - datetime_object).total_seconds()
                                print(total_seconds)

                                if total_seconds > 3600:
                                    ref = db.reference(f'Students/{id}')
                                    result_student_info['total_attendance'] += 1
                                    ref.child('total_attendance').set(result_student_info['total_attendance'])
                                    ref.child('last_attendance').set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                else:
                                    modetype = 3
                                    counter = 0
                                    imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]

                            if modetype != 3:
                                if 10 < counter <= 20:
                                    modetype = 2
                                imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]

                                if counter < 10:
                                    cv2.putText(imgbackground, str(result_student_info['total_attendance']), (891, 113),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                                    cv2.putText(imgbackground, str(result_student_info['name']), (958, 420),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                                    cv2.putText(imgbackground, str(result_student_info['major']), (1015, 577),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                                    cv2.putText(imgbackground, str(id), (970, 490),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                                    imgbackground[175:175 + 216, 909:909 + 216] = result_imgstud

                                counter += 1
                                if counter >= 30:
                                    counter = 0
                                    modetype = 0
                                    student_info = []
                                    result_imgstud = []
                                    imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
                    else:
                        counter = 0
                        modetype = 0
                    cv2.imshow('background', imgbackground)
                    cv2.waitKey(1)
                elif class_name[cls]=='device' or  'mask' or 'photo' and conf>0.3:
                    print("error")
                    modetype = 4
                    counter=0
                    imgbackground[162:162 + 480, 55:55 + 640] = img
                    imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
                    cv2.putText(imgbackground[44:44 + 633, 808:808 + 414], f'{class_name[cls]}', (167, 340),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('background', imgbackground)
                    cv2.waitKey(1)

# تشغيل البرنامج غير المتزامن
asyncio.run(main())


