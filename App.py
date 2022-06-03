import cv2 as cv
import sqlite3



# Initialization value
scaleFactor = 1.3
minNeighbor = 5
pathHaarcascade = 'haarcascade_frontalface_default.xml'
windowName = 'Face Recognition'
pathDestination = 'Dataset'
pathTraining = 'Training/training.yml'
fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,255)


#Variable
videoCapture = cv.VideoCapture(0)
cascadeClassifier = cv.CascadeClassifier(pathHaarcascade)
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(pathTraining)

def getProfile(id):
    connect2SQLlite = sqlite3.connect("YOUR DATABASE")
    getUserData = "SELECT * FROM nameTable WHERE id = " + str(id)
    cursor = connect2SQLlite.execute(getUserData)
    profile = None

    for rowData in cursor:
        profile = rowData
    connect2SQLlite.close()
    return profile

a=0
id = 0
while True:
    a += 1
    check,frame = videoCapture.read()
    makeGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faceDetection = cascadeClassifier.detectMultiScale(makeGray,scaleFactor,minNeighbor)

    for (x,y,w,h) in faceDetection:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = recognizer.predict(makeGray[y:y+h,x:x+w])

        userProfile = getProfile(id)
        if (userProfile != None):
            cv.putText(frame,str(userProfile[1]),(x+w,y+h+30),fontFace,fontScale,fontColor,1)
            cv.putText(frame,str(userProfile[2]),(x+w,y+h+60),fontFace,fontScale,fontColor,1)
            cv.putText(frame,str(userProfile[3]),(x+w,y+h+90),fontFace,fontScale,fontColor,1)

    cv.imshow(windowName,frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
