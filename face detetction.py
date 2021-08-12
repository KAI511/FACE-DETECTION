import cv2
import matplotlib.pyplot as plt

#img = cv2.imread(r"C:\Users\kaila\Desktop\text files\pic\11.jpg")
# img.shape  # use pycharm n enefit of vscode over ? n how to write in it
# img[0]
# plt.imshow(img)
#  #plt.waitforbuttonpress() #show fro plot to wait on screen


haar_data = cv2.CascadeClassifier(r'E:\code\pythonProject1\haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        cv2.imshow('result', img)
        # 27 is ASCII of escape
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()





