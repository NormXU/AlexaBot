#!/usr/bin/env python


import rospy
import cv2
import cv2 as cv
from rbx1_vision.ros2opencv2 import ROS2OpenCV2
import numpy as np

class HandDetect(ROS2OpenCV2):
    def __init__(self, node_name): 
        super(HandDetect, self).__init__(node_name)
          
        # Do we show text on the display?
        self.show_text = rospy.get_param("~show_text", True)
        
        # How big should the feature points be (in pixels)?
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        # Good features parameters
        self.gf_maxCorners = rospy.get_param("~gf_maxCorners", 200)
        self.gf_qualityLevel = rospy.get_param("~gf_qualityLevel", 0.02)
        self.gf_minDistance = rospy.get_param("~gf_minDistance", 7)
        self.gf_blockSize = rospy.get_param("~gf_blockSize", 10)
        self.gf_useHarrisDetector = rospy.get_param("~gf_useHarrisDetector", True)
        self.gf_k = rospy.get_param("~gf_k", 0.04)
        
        # Store all parameters together for passing to the detector
        self.gf_params = dict(maxCorners = self.gf_maxCorners, 
                       qualityLevel = self.gf_qualityLevel,
                       minDistance = self.gf_minDistance,
                       blockSize = self.gf_blockSize,
                       useHarrisDetector = self.gf_useHarrisDetector,
                       k = self.gf_k)

        # Initialize key variables
        self.keypoints = list()
        self.detect_box = None
        self.mask = None
        self.HandImage = None
        
 # Cover the function which has the same name as processing and redefine the details       
    def processing(self, cv_image):
        try:
            # If the user has not selected a region, just return the image
            if not self.detect_box:
                return cv_image

            cv2.rectangle(frame,(170,170),(400,400),(0,255,0),0)
            crop_image = frame[170:400, 170:400]
            height, width, channels = crop_image.shape

            blur = cv2.blur(crop_image,(3,3), 0)

            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            lower_range = np.array([2,0,0])
            upper_range = np.array([16,255,255])

            mask = cv2.inRange(hsv,lower_range,upper_range)

            skinkernel = np.ones((5,5))
            dilation = cv2.dilate(mask, skinkernel, iterations = 1)
            erosion = cv2.erode(dilation, skinkernel, iterations = 1) 

            iltered = cv2.GaussianBlur(erosion, (15,15), 1)
            ret,thresh = cv2.threshold(filtered, 127, 255, 0)

            
            # Process any special keyboard commands
            if self.keystroke != -1:
                try:
                    cc = chr(self.keystroke & 255).lower()
                    if cc == 'c':
                        # Clear the current keypoints
                        keypoints = list()
                        self.detect_box = None
                except Exception as e:
                    print e
        except Exception as e:
            print e
                                
        return hand_img

    def hand_finding(self, input_image, detect_box):

    	blur = cv2.blur(input_image,(3,3))
    	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    	kernel_square = np.ones((11,11),np.uint8)
    	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    	dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    	erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    	dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    	filtered = cv2.medianBlur(dilation2,5)
    	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    	dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    	dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    	median = cv2.medianBlur(dilation2,5)
    	ret,thresh = cv2.threshold(median,127,255,0)

    	return thresh


if __name__ == '__main__':
    try:
        node_name = "hand_detect"
        HandDetectObject= HandDetect(node_name)
        
        while not rospy.is_shutdown():
            if HandDetectObject.display_image is not None:
                #HandDetectObject.image_show(HandDetectObject.cv_window_name, HandDetectObject.HandImage)
                #cv2.imshow("Hand Detected", HandDetectObject.HandImage)

    except KeyboardInterrupt:
        print "Shutting down the Good Features node."
        cv.DestroyAllWindows()

