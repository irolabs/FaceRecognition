import cv2 as cv
import os
import numpy as np
from PIL import Image


# Initialization value
pathHaarcascade = 'haarcascade_frontalface_default.xml'
windowName = 'Face Recognition'
pathDestination = 'Dataset'

# Variable
recognizer = cv.face.LBPHFaceRecognizer_create()
cascadeClassifier = cv.CascadeClassifier(pathHaarcascade)


def getImageWithLabel(path):
    #getting image path
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]
    #save into array
    sampleFace = []
    someId = []
    #looping
    for imgpath in imagePath:
        imagePIL = Image.open(imgpath).convert('L')
        npImage = np.array(imagePIL,'uint8')

        id = int(os.path.split(imgpath)[-1].split(".")[1])

        faceDetection = cascadeClassifier.detectMultiScale(npImage)

        for (x,y,w,h)in faceDetection:
            sampleFace.append(npImage[y:y+h,x:x+w])
            someId.append(id)
    return sampleFace, someId

faceDetection,someId = getImageWithLabel(pathDestination)
recognizer.train(faceDetection,np.array(someId))
recognizer.save('Training/training.yml')





