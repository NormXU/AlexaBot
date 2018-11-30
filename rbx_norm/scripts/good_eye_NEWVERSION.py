#!/usr/bin/env python


import rospy
import cv2
import cv2 as cv
from rbx1_vision.ros2opencv2 import ROS2OpenCV2
import numpy as np
import math
from keras import backend as K
from std_msgs.msg import Bool
import Matrix_CV_ML3D as DImage
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10



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
        self.hand_flag = None
        self.boolFwdFlag = Bool()
        self.boolRightFlag = Bool()
        self.boolLeftFlag = Bool()
        self.boolTurnFlag = Bool()
        self.count = 0
        self.eyecount = 0

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
 # Cover the function which has the same name as processing and redefine the details       
    def processing(self, cv_image):
        try:
            # If the user has not selected a region, just return the image
            cv_image = cv2.flip(cv_image, 1)
            if not self.detect_box:
                return cv_image

            self.boolFwdFlag.data = 0
            self.boolTurnFlag.data = 0
            self.boolRightFlag.data = 0
            self.boolLeftFlag.data = 0


            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if np.size(faces)==0:
            
                print ('stop')
                cv2.putText(cv_image,"stop", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

            for (x,y,w,h) in faces:
                ##########
                num_face = np.size(faces)
                if np.size(faces)>1:
                    cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = cv_image[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray,1.1,10)
                if np.size(eyes)==0:
                    print ('stop')
                    cv2.putText(cv_image,"stop", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                num_eyes = np.size(eyes)
                self.eyecount = self.eyecount+1
                num_eyes = num_eyes/4
                if num_eyes == 1:
                    if eyes[0,0]<faces[0,2]/2:
                        print ('move right!')
                        cv2.putText(cv_image,"move right!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        self.boolRightFlag.data = 1
                    else:
                        print ('move left!')
                        cv2.putText(cv_image,"move left!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        self.boolLeftFlag.data = 1
                if num_eyes == 2:
                    print ('move forward!')
                    cv2.putText(cv_image,"move forward!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    self.boolFwdFlag.data = 1
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #cv2.imshow('frame1',cv_image)


               

            
        except Exception as e:
            print e
            
                                
        return cv_image


if __name__ == '__main__':
    try:
        node_name = "eye_tracking"
        goodfeatures= GoodFeatures(node_name)
        #hand_flag = None
        #pub = rospy.Publisher('hand_detect_forward', Bool, queue_size=10)
        
        #rate = rospy.Rate(10) # 10hz
        
        while not rospy.is_shutdown():
            if goodfeatures.display_image is not None:
                goodfeatures.image_show(goodfeatures.cv_window_name, goodfeatures.display_image)

            hand_pub_fwd = rospy.Publisher("/Alexa/fwd_cmd_realtime", Bool, queue_size=1)
            hand_pub_fwd.publish(goodfeatures.boolFwdFlag)

            hand_pub_right = rospy.Publisher("/Alexa/right_cmd_realtime", Bool, queue_size=1)
            hand_pub_right.publish(goodfeatures.boolRightFlag)

            hand_pub_left = rospy.Publisher("/Alexa/left_cmd_realtime", Bool, queue_size=1)
            hand_pub_left.publish(goodfeatures.boolLeftFlag)
                
    except KeyboardInterrupt:
        print "Shutting down the Good Features node."
        cv.DestroyAllWindows()


