import cv2

bird1_cascade = cv2.CascadeClassifier('bird1-cascade.xml')

cap = cv2.VideoCapture(0)

while 1:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    birds1 = bird1_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in birds1:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Bird', (x-y, y-h), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()