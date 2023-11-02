import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('DOC_PIC.JPEG')
greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordenates = trained_face_data.detectMultiScale(greyscale_image)
#print(face_coordenates)
(x,y,w,h) = face_coordenates[0]
#print(face_coordenates[0])
cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey( )

print('Code Completed')
