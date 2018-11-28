#!/usr/bin/env python
import dlib
from skimage import io
import rospy
import cv2
import cv2 as cv
from rbx1_vision.ros2opencv2 import ROS2OpenCV2
import numpy as np
import math
import pygame
from std_msgs.msg import Bool

class GoodFeatures(ROS2OpenCV2):
    def __init__(self, node_name): 
        super(GoodFeatures, self).__init__(node_name)
          
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
        self.boolFlag = Bool()
        #self.detector1 = dlib.fhog_object_detector("Pedestriandetector.svm")
        #self.detector2 = dlib.fhog_object_detector("Pedestriandetector2.svm")
        self.detector3 = dlib.fhog_object_detector("Pedestriandetector33.svm")
        self.detectors = [self.detector3]
        pygame.init()
        self.s = pygame.mixer.Sound('Dong.WAV')
        
 # Cover the function which has the same name as processing and redefine the details       
    def processing(self, cv_image):
        try:
            # If the user has not selected a region, just return the image
            cv_image = cv2.flip(cv_image, 1)
            if not self.detect_box:
                return cv_image

            #detector = dlib.simple_object_detector("Pedestriandetector.svm")
            self.boolFlag.data = 0

            scale_percent = 70 # percent of original size
            width = int(cv_image.shape[1] * scale_percent / 100)
            height = int(cv_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            cv_image = cv2.resize(cv_image,dim,interpolation = cv2.INTER_AREA)
    		#cv_image = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(self.detectors, cv_image, upsample_num_times=1, adjust_threshold=0.0)
            for d in boxes:
    			cv2.rectangle(cv_image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
    			self.boolFlag.data = 1
    			self.s.play()

            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            
        except Exception as e:
            print e
            
                                
        return cv_image



if __name__ == '__main__':
    try:
        node_name = "hog_svm"
        goodfeatures= GoodFeatures(node_name)

        
        while not rospy.is_shutdown():
            if goodfeatures.display_image is not None:
                goodfeatures.image_show(goodfeatures.cv_window_name, goodfeatures.display_image)


            hand_pub_fwd = rospy.Publisher("/Alexa/find_people", Bool, queue_size=1)
            hand_pub_fwd.publish(goodfeatures.boolFlag)

                
    except KeyboardInterrupt:
        print "Shutting down the Good Features node."
        cv.DestroyAllWindows()

