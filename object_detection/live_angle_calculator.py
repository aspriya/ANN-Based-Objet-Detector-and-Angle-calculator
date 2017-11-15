import cv2
import numpy as np
import math
 
#uses distance formula to calculate distance
def distance(x1, y1, x2,y2):
  dist = math.sqrt((math.fabs(x2-x1))**2 + ((math.fabs(y2-y1)))**2)
  return dist
 
#filters for green color and returns green color position.
def findgreen(frame):
  maxcontour = None
  green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  greenlow = np.array([30,40,187])#replace with your HSV Values
  greenhi = np.array([75,255,255])#replace with your HSV Values
  mask = cv2.inRange(green, greenlow, greenhi)
  res = cv2.bitwise_and(frame, frame, mask=mask)

  img, cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  if len(cnts) > 0:
    maxcontour = max(cnts, key = cv2.contourArea)
 
    #All this stuff about moments and M['m10'] etc.. are just to return center coordinates
    M = cv2.moments(maxcontour)
    if M['m00'] > 0 and cv2.contourArea(maxcontour) > 2000:
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return (cx, cy), True
    else:
      #(700,700), arbitrary random values that will conveniently not be displayed on screen
      return (700,700), False
  else:
    return (700,700), False
 
#filters for orange color and returns orange color position.
def findorange(frame):
  maxcontour = None
  orange = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  orangelow =  np.array([118, 88, 125])#replace with your HSV Values
  orangehi = np.array([179, 255, 255])#replace with your HSV Values
  mask = cv2.inRange(orange, orangelow, orangehi)
  res = cv2.bitwise_and(frame, frame, mask=mask)

  img, cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  if len(cnts) > 0:
    maxcontour = max(cnts, key = cv2.contourArea)
    M = cv2.moments(maxcontour)
    if M['m00'] > 0 and cv2.contourArea(maxcontour)>2000:
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return (cx, cy), True
    else:
      return (700,700), False
  else:
    return (700,700), False


#capture video
cap = cv2.VideoCapture(0)
 
while(1):
  _, frame = cap.read()
  fra = frame.copy() #if you're sending the whole frame as a parameter,easier to debug if you send a copy
 
  #get coordinates of each object
  (greenx, greeny), glogic = findgreen(fra)
  (orangex, orangey), ologic = findorange(fra)
  #draw two circles around the objects (you can change the numbers as you like)
  cv2.circle(frame, (greenx, greeny), 20, (0, 255, 0), -1)
  cv2.circle(frame, (orangex, orangey), 20, (0, 128, 255), -1)

  if glogic and ologic:
    #quantifies the hypotenuse of the triangle
    hypotenuse =  distance(greenx,greeny, orangex, orangey)
    #quantifies the horizontal of the triangle
    horizontal = distance(greenx, greeny, orangex, greeny)
    #makes the third-line of the triangle
    thirdline = distance(orangex, orangey, orangex, greeny)
    #calculates the angle using trigonometry
    angle = np.arcsin((thirdline/hypotenuse))* 180/math.pi
 
    #draws all 3 lines
    cv2.line(frame, (greenx, greeny), (orangex, orangey), (0, 0, 255), 2)
    cv2.line(frame, (greenx, greeny), (orangex, greeny), (0, 0, 255), 2)
    cv2.line(frame, (orangex,orangey), (orangex, greeny), (0,0,255), 2)

    #Allows for calculation until 180 degrees instead of 90
    if orangey < greeny and orangex > greenx:
      cv2.putText(frame, str(int(angle)), (greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
    elif orangey < greeny and orangex < greenx:
      cv2.putText(frame, str(int(180 - angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
    elif orangey > greeny and orangex < greenx:
      cv2.putText(frame, str(int(180 + angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
    elif orangey > greeny and orangex > greenx:
      cv2.putText(frame, str(int(360 - angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128, 229), 2) 
    
  cv2.imshow('Angle', frame)
  k = cv2.waitKey(5) & 0xFF
  if k == ord('q'):
    break
