import cv2 as cv


# Initialization value
scaleFactor = 1.3
minNeighbor = 5
pathHaarcascade = 'haarcascade_frontalface_default.xml'
windowName = 'Face Recognition'
pathDestination = 'Dataset/User.'

# Variable
vidCapture = cv.VideoCapture(0)
cascadeClassifier = cv.CascadeClassifier(pathHaarcascade)
a = 0
id = input('Input UserId: ')

while True:
    a += 1
    check,frame = vidCapture.read()
    makeGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    faceDetection = cascadeClassifier.detectMultiScale(makeGray,scaleFactor,minNeighbor)

    for (x,y,w,h) in faceDetection:
        cv.imwrite(pathDestination+str(id)+"."+str(a)+".jpg",makeGray[y:y+h,x:x+h])
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow(windowName,frame)
    #key = cv.waitKey(1)
    if (a>49):
        break
    print(a)

vidCapture.release()
