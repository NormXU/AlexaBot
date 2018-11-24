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
        self.model = load_model('model.h5')
        print('test model...')
        #print self.model.predict(np.zeros((1, len(word_index)+1)))
        #https://www.jianshu.com/p/c84ae0527a3f
        print(self.model.predict(np.zeros((1,3,50,65))))
        print('test Done !')

        
 # Cover the function which has the same name as processing and redefine the details       
    def processing(self, cv_image):
        try:
            # If the user has not selected a region, just return the image
            cv_image = cv2.flip(cv_image, 1)
            if not self.detect_box:
                return cv_image
    

            cv2.rectangle(cv_image,(170,170),(400,400),(0,255,0),0)
            crop_image = cv_image[170:400, 170:400]
            #crop_image = cv_image
            height, width, channels = crop_image.shape

            blur = cv2.blur(crop_image,(5,5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            lower_range = np.array([2,0,0])
            #upper_range = np.array([10,255,255])
            upper_range = np.array([16,255,255])

            mask = cv2.inRange(hsv,lower_range,upper_range)
            skinkernel = np.ones((5,5))

            dilation = cv2.dilate(mask, skinkernel, iterations = 2)
            erosion = cv2.erode(dilation, skinkernel, iterations = 2)

            filtered = cv2.GaussianBlur(dilation, (15,15), 1)
            ret,thresh = cv2.threshold(filtered, 127, 255, 0)

            Label_ret, markers = cv2.connectedComponents(thresh)
            num = markers.max()
            #print(num)

            for i in range(1, num+1):
                pts =  np.where(markers == i)
                if len(pts[0]) < 200:
                    markers[pts] = 0

            label_hue = np.uint8(markers.copy())
            label_hue = np.where(label_hue != 0, 255, label_hue)

            resColored = cv2.bitwise_and(crop_image,crop_image,mask = label_hue)
            res = cv2.cvtColor(resColored, cv2.COLOR_BGR2GRAY)


            # CNN Predict
            img = cv2.resize(resColored,(50,65),interpolation = cv2.INTER_CUBIC)
            img = np.reshape(img,(1,3,50,65))
            pred  = self.model.predict(img)
            predicted = np.argmax(pred,axis=1)
            #print(predicted)




            #find contours
            _,contours,hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            #find contour of max area(hand)
            if (len(contours) > 0):
                cnt = max(contours, key = lambda x: cv2.contourArea(x))

                #approx the contour a little
                epsilon = 0.0005*cv2.arcLength(cnt,True)
                approx= cv2.approxPolyDP(cnt,epsilon,True)

                #make convex hull around hand
                hull = cv2.convexHull(cnt)

                #define area of hull and area of hand
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)


                #find the percentage of area not covered by hand in convex hull
                arearatio=((areahull-areacnt)/areacnt)*100

                #find the defects in convex hull with respect to hand
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)

                # l = no. of defects
                l=0

                self.boolFwdFlag.data = 0
                self.boolTurnFlag.data = 0
                self.boolRightFlag.data    = 0
                self.boolLeftFlag.data = 0


                #code for finding no. of defects due to fingers
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    pt= (100,180)

                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

                    #distance between point and convex hull
                    d=(2*ar)/a
                    #print(a)


                    # apply cosine rule here

                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                    if angle <= 90:
                        l += 1
                    #cv2.circle(crop_image, far, 3, [255,0,0], -1)
                    #draw lines around hand
                    cv2.line(crop_image,start, end, [0,255,0], 2)
            
            
                    #l+=1
        
                    #print corresponding gestures which are in their ranges
                    #print(l)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if l == 4 and predicted == 1:
                        cv2.putText(cv_image,"Go Forward", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        
                        self.boolFwdFlag.data = 1

                    if l!=5 and predicted == 3 and l!=0:
                    	cv2.putText(cv_image,"Turn Left", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    	
                    	self.boolLeftFlag.data = 1

                    if l!=5 and predicted == 2 and l!=0:
                    	cv2.putText(cv_image,"Turn Right", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    	self.boolRightFlag.data = 1

            
        except Exception as e:
            print e
            
                                
        return cv_image


if __name__ == '__main__':
    try:
        node_name = "hand_gesture"
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

