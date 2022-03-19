import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
import cv2


def yolo():
        config_file_path = "custom-yolov4-tiny-detector.cfg"
        weight_file_path = "custom-yolov4-tiny-detector_last.weights"
        board=Darknet(config_file_path,inference=True)
        board.load_weights(weight_file_path)
        board.cuda()
        return board

class angle_detection():
    

    def my_detect(m,cv_img):
        use_cuda=True
        img=cv2.resize(cv_img, (m.width, m.height))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = do_detect(m, img, 0.2, 0.6, use_cuda)
        if len(boxes[0])==0:
            return [False,0,0,0,0]
        box=boxes[0][0]
        h, w, _ = cv_img.shape
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        return [True,x1,y1,x2,y2]

    def highpass(img, sigma):
        return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

    def depth_data_to_edge_pass(depth_data):
        b = np.load(depth_data)          
        b = b-np.min(b)
        b = b/np.max(b)
        b = 255*b
        b = np.floor(b)

        aperture_size = 5
        img = np.uint8(b)
        edges = cv2.Canny(img, 1, 100, apertureSize=aperture_size)
        edge_pass = angle_detection.highpass(edges, 3) 
        plt.imshow(edge_pass, cmap = 'gray')
        cv2.imwrite("depth_edge.png", edge_pass)
    
    def get_bbox_coords(depth_data, model):          
        angle_detection.depth_data_to_edge_pass(depth_data)
        yolo_inp_img = cv2.imread('depth_edge.png')

        frame = yolo_inp_img.copy()
        ret,x1,y1,x2,y2 = angle_detection.my_detect(board,frame)      

        if ret :
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3) 
        
        cv2.imshow("Bbox frame", frame)
        cv2.waitKey(0)
        tolerance = 15   
        cropped_image = yolo_inp_img[y1-tolerance:y2+tolerance, x1-tolerance:x2+tolerance]
        
        img = cropped_image.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 2)
        
        cntrs, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(cntrs)
        
        cv2.drawContours(cropped_image, cntrs, -1, (0, 0, 255), thickness=3)
        cntrs = sorted(cntrs, key=lambda x: cv2.contourArea(x), reverse=True)
        rotrect = cv2.minAreaRect(cntrs[0])
        box = cv2.boxPoints(rotrect)
        box = np.int0(box)
        result = img.copy()
        cv2.drawContours(result,[box],0,(0,0,255),2)
        angle = rotrect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        print(angle,"deg")
        cv2.imwrite("final.png", result)
        cv2.imshow("RESULT", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       
    
if __name__ == "__main__":
    board = yolo()
    angle_detection.get_bbox_coords("./depth/depth_camera_image106.npy", board)