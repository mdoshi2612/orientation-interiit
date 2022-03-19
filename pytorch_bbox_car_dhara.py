import os
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import math
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
import time
import cv2

weight_file_path = "/home/vivek/Desktop/yolo/custom-yolov4-tiny-detector_last.weights"
config_file_path = "/home/vivek/Desktop/yolo/custom-yolov4-tiny-detector.cfg"

def yolo():
    board=Darknet(config_file_path,inference=True)
    board.load_weights(weight_file_path)
    board.cuda()
    return board


def my_detect(m,cv_img):
    use_cuda=True
    img=cv2.resize(cv_img, (m.width, m.height))
    # print(img.shape)
    # print(img)
    # img = np.array(img, dtype=np.int16)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, img, 0.2, 0.6, use_cuda)
    if len(boxes[0])==0:
        return [False,0,0,0,0]
    box=boxes[0][0]
    h,w,c=cv_img.shape
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return [True,x1,y1,x2,y2]

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

def depth_data_to_edge_pass():
    b = np.load("depth_camera_image10.npy")          
    b = b-np.min(b)
    b = b/np.max(b)
    b = 255*b
    b = np.floor(b)

    aperture_size = 5
    img = np.uint8(b)
    edges = cv2.Canny(img,1,100,apertureSize=aperture_size)

    high_pass = highpass(img, 3)
    edge_pass = highpass(edges, 3)  #numpy array
    plt.imshow(edge_pass, cmap = 'gray')
    cv2.waitKey(0)
    plt.savefig('depth_edge.png')
    
def get_bbox_coords():          
    board = yolo()
    depth_data_to_edge_pass()
    yolo_inp_img = cv2.imread('depth_edge.png')     
    yolo_inp_img = yolo_inp_img[33:252, 72:366]     
    #cv2.imshow ("ok" , yolo_inp_img)
    #cv2.waitKey(0)
    frame = yolo_inp_img
    ret,x1,y1,x2,y2 = my_detect(board,frame)        
    car_img = yolo_inp_img[x1:x2, y1:y2] 
    # cv2.imshow("cropped" , car_img)
    # cv2.waitKey(0)

    print ("image")
    if ret :
      frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
      #car_img = yolo_inp_img[x1:x2, y1:y2] 
      cv2.imwrite("car_cropped.png" , car_img)
    # plt.imshow(frame),plt.show()

    # cv2.imshow("hi", car_img )
    # cv2.waitKey(0)
    return car_img      

def outer_countouring():
    img = cv2.imread("car_cropped.png")
    hh, ww, cc = img.shape

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image
    ret, thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)

    cv2.imshow("Image", thresh)
    cv2.waitKey(0)
    # find outer contour
    cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # get rotated rectangle from outer contour
    rotrect = cv2.minAreaRect(cntrs[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    # draw rotated rectangle on copy of img as result
    result = img.copy()
    cv2.drawContours(result,[box],0,(0,0,255),2)

    # get angle from rotated rectangle
    angle = rotrect[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    print(angle,"deg")

    # write result to disk
    cv2.imwrite("wing2_rotrect.png", result)

    #cv2.imshow("THRESH", thresh)
    cv2.imshow("RESULT", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return



get_bbox_coords()
outer_countouring()