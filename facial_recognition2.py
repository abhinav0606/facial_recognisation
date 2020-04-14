import cv2
import numpy as np
import os
data_path="/root/handshakes/"
only_files=[f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]
t=[]
l=[]
for i,files in enumerate(only_files):
    image_path=data_path+only_files[i]
    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    t.append(np.asarray(image,dtype=np.uint8))
    l.append(i)
l=np.asarray(l,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(t),np.asarray(l))
face_classifier=cv2.CascadeClassifier('/root/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def face_determiner(img,size=0.5):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grey,1.3,5)
    if faces is ():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    img,face=face_determiner(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        confidence=0
        if result[1]<500:
            confidence=int(100*((1-result[1]/300)))
        cv2.putText(img,str(confidence)+"% face matched",(300,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        if confidence>75:
            cv2.putText(img,"unlocked",(400,450),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 255),2)
            cv2.imshow("Cropped",img)
        else:
            cv2.putText(img,"locked", (400,450), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,0),2)
            cv2.imshow("Cropped",img)
    except:
        # cv2.putText(img,"Not Found", (400, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        # cv2.imshow("Cropped",img)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()