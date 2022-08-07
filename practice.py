import cv2
import numpy as np

img=cv2.imread("E:\Documents\FTE\MSD\Work\Banana-Single.jpg")
img=cv2.resize(img,(400,400))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV)
dilate = cv2.dilate(thresh,(1,1),iterations = 6)
frame = cv2.bitwise_and(img,img,mask=dilate)
cnts, heir = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    epsilon = 0.0001*cv2.arcLength(c,True)
    data= cv2.approxPolyDP(c,epsilon,True)
    hull = cv2.convexHull(data)
    cv2.drawContours(frame, [c], -1, (255,0,0), 2)
    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
    hull2 = cv2.convexHull(c,returnPoints = False)
    defect = cv2.convexityDefects(c,hull2)

for i in range(defect.shape[0]):
    s,e,f,d = defect[i,0]
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    cv2.circle(frame,far,5,[0,0,255],-1)

cv2.imshow("Original",img)
cv2.imshow("Gray",gray)
cv2.imshow("mask",thresh)
cv2.imshow("Contours",frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
