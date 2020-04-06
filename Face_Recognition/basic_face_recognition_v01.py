import cv2, time

face_casc=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # import xlm for classifiers
video=cv2.VideoCapture(0) #activate camera

a=1
while True:
    a=a+1
    check, frame = video.read() #check is boolean, frame is numpy array , video.read() returns a true if video is running and frame

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert to grayscale as its better for image recognition

    faces=face_casc.detectMultiScale(gray, #use Cascade classifier to detect faces in gray image
    scaleFactor=1.2, #value of accuracy: 1.05 scales the image by 5 percent down each itteration, the higher the value the less accurate
    minNeighbors=5
    )
    for x,y,width,height in faces: #iterate through all values of faces
        frame=cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0), 3) # draw rectangle with arguments: background file, (startinpoint x, startingpoint y), (endpoint x, endpoint y), RGB color value, border width

    cv2.imshow("Face recognition", frame)

    key=cv2.waitKey(1)
    if key ==ord("q"):
        break

video.release()
cv2.destroyAllWindows()
