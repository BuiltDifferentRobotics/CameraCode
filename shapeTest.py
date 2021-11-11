import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([25,100,50])
    upper_red = np.array([100,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res[res != 0] = 255

    # ksize
    ksize = (20, 20)
  
    # Using cv2.blur() method 
    res = cv2.blur(res, ksize) 

    img = res
  
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 180, cv2.THRESH_BINARY)
  
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    i = 0

    if len(contours)<1:
        continue

    maxcontour = max(contours, key=cv2.contourArea)
  
    # list for storing names of shapes
    
    contour = maxcontour # only do the max contour

    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # finding center point of shape
    M = cv2.moments(contour)
    x = 0
    y = 0
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    elif len(approx) == 4:
        cv2.putText(img, 'Quadrilateral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        cv2.putText(img, 'circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center = (x, y)
    wh = (w, h)

    # at 20 inches away, w = 113
    dist = 20*(113/w)


    refrect = (287+w/2, 214+h/2, 63, 49)
    refarea = refrect[2] * refrect[3]
    print("center " ,center, " wh", wh)
    print("dist: ", dist)


    '''
    nx= (1/960) * (x-959.5)
    ny= (1/540) * (y-539.5)

    horizontol_fov = 1.117
    vertical_fov = 0.7854

    vpw = 2.0*math.tan(horizontol_fov/2)
    vph = 2.0*math.tan(vertical_fov/2)
    
    x = vpw/2 * nx
    y = vph/2 * ny

    ax = math.atan2(1,x)
    ay = math.atan2(1,y)
    print("anglex:", ax, "angley:", ay)
    '''

    
    # displaying the image after drawing contours
    cv2.imshow('shapes', img)
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()