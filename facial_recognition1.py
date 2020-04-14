import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier('/root/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def face_extractor(img):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grey,1.2,5)
    if faces is ():
        return None
    for(x,y,w,h) in faces:
        cropped=img[y:y+h,x:x+w]
    return cropped
cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count=count+1
        face=cv2.resize(face_extractor(frame),(400,400))
        face=cv2.cvtColor(face_extractor(frame),cv2.COLOR_BGR2GRAY)
        file_name="/root/handshakes/user"+str(count)+'.JPG'
        cv2.imwrite(file_name,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow("Cropped_face",face)
    else:
        print("not Found")
    if cv2.waitKey(1)==13 or count==100:
        break
cap.release()
cv2.destroyAllWindows()
