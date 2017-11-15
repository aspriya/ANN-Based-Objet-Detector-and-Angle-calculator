# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim  
# Run above in each shell in order to run this script

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

cap = cv2.VideoCapture(0)
 

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")



from utils import label_map_util
from utils import visualization_utils as vis_util


#CODES TO GET THE TRAINED MODEL
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90



# ANGLE CALCULATION FUNCTIONS

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
 
    #these are to return coordinates of centerare just to return center coordinates
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

#END OF ANGLE CALCULATION FUNCTIONS



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    

    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    
    taco = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
    print(taco)
    for a in taco:
      # ANGLE IS CALCUALTED IF IT DETECTS A BOTTLE OR SPOON ONLY.
      if a['name'] == "bottle" or a['name'] == 'spoon':
        print "abc"
        
        #CALLING OPENCV AND ANGLE CALCULATION FUNCTIONS TO CALCULATE ANGLE OF CAPTURED IMAGE. 
        
        fra = image_np.copy() #if you're sending the whole frame as a parameter,easier to debug if you send a copy
 
        #get coordinates of each object
        (greenx, greeny), glogic = findgreen(fra)
        (orangex, orangey), ologic = findorange(fra)
        #draw two circles around the objects (you can change the numbers as you like)
        cv2.circle(image_np, (greenx, greeny), 20, (0, 255, 0), -1)
        cv2.circle(image_np, (orangex, orangey), 20, (0, 128, 255), -1)

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
          cv2.line(image_np, (greenx, greeny), (orangex, orangey), (0, 0, 255), 2)
          cv2.line(image_np, (greenx, greeny), (orangex, greeny), (0, 0, 255), 2)
          cv2.line(image_np, (orangex,orangey), (orangex, greeny), (0,0,255), 2)

          #Allows for calculation until 180 degrees instead of 90
          if orangey < greeny and orangex > greenx:
            cv2.putText(image_np, str(int(angle)), (greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
          elif orangey < greeny and orangex < greenx:
            cv2.putText(image_np, str(int(180 - angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
          elif orangey > greeny and orangex < greenx:
            cv2.putText(image_np, str(int(180 + angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128,220), 2)
          elif orangey > greeny and orangex > greenx:
            cv2.putText(image_np, str(int(360 - angle)),(greenx-30, greeny), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,128, 229), 2)


        #END OF CALLING OPENCV AND ANGLE CALCULATION FUNCTIONS TO CALCULATE ANGLE OF CAPTURED IMAGE.
    
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    k = cv2.waitKey(0)
      
  
