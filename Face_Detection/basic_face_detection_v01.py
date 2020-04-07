"""A basic movement- and face detection approach implementd with openCV2"""
import cv2, pandas, time
from datetime import datetime


status_list=[0,0]
times=[]
frame1=None
df=pandas.DataFrame(columns=["Start_Time","End_Time"])
face_casc=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # import xlm for classifiers

video=cv2.VideoCapture(0) #activate camera


while True:
    check, frame = video.read() #check is boolean, frame is numpy array , video.read() returns a true if video is running and frame provides the image
    motion_status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to grayscale as its better for image recognition
    gray=cv2.GaussianBlur(gray,(21,21),0) #blurring image to smooth out details

    if frame1 is None:
        frame1 = gray
        continue

    ### Create Delta/ Difference Frame for motion detection    ### Image threshold simplification  Motion detection
    delta_frame=cv2.absdiff(frame1,gray) #bright =motion, dark =less motion
    delta_threshold=cv2.threshold(delta_frame, 80, 255, cv2.THRESH_BINARY)[1] #returns 2 values: 1. suggesting value for threshold, 2. actual frame; attributes: valueimage, threshold and color assigned to moving areas, method
    delta_threshold=cv2.dilate(delta_threshold, None, iterations=2)

    cnts, _ =cv2.findContours(delta_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detecting contours change = detecting motion

    for contour in cnts:
        if cv2.contourArea(contour) <10000: #less than 8000 pixels in area
            continue

        motion_status=1 #setting motion status to one
        x,y,w,h = cv2.boundingRect(contour)
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 1) # draw rectangle with arguments: background file, (startinpoint x, startingpoint y), (endpoint x, endpoint y), RGB color value, border width

    status_list.append(motion_status)
    status_list=status_list[-2:] #keep only last 2 items of the status list in order to save memory

    ### Face Recognition###
    faces=face_casc.detectMultiScale(gray, #use Cascade classifier to detect faces in gray image
    scaleFactor=1.1, #value of accuracy: 1.05 scales the image by 5 percent down each itteration, the higher the value the less accurate
    minNeighbors=5
    )

    # add frame around detected face
    for x,y,width,height in faces: #iterate through all values of faces
        frame=cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0), 3) # draw rectangle with arguments: background file, (startinpoint x, startingpoint y), (endpoint x, endpoint y), RGB color value, border width


    ### Display Results
    cv2.imshow("Delta", delta_threshold)
    cv2.imshow("Face recognition", frame) #title and source

    ### Record motion
    if status_list[-1] == 1 and status_list[-2] ==0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] ==1:
        times.append(datetime.now())

    ### Video & Window Controls
    key=cv2.waitKey(1) #wait for 1 mil sec before image refresh
    if key ==ord("q"): #if push q key break while loop and execute below commands (close camera and destroy widow)
        if motion_status==1:
            times.append(datetime.now())
        break
# end of while loop


### Time tracking of movement - writing file

for i in range(0,len(times),2):
    df=df.append({"Start_Time":times[i],"End_Time":times[i+1]}, ignore_index=True)

df.to_csv("Times.csv", index=False, header=True)

video.release()
cv2.destroyAllWindows()
