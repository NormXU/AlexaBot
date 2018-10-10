#!/usr/bin/env python


import rospy
import cv2
import cv2 as cv
from rbx1_vision.ros2opencv2 import ROS2OpenCV2
import numpy as np
import math
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
        self.hand_flag = None
        self.boolFlag = Bool()
        
 # Cover the function which has the same name as processing and redefine the details       
    def processing(self, cv_image):
        try:
            # If the user has not selected a region, just return the image
            if not self.detect_box:
                return cv_image
    

            cv2.rectangle(cv_image,(170,170),(400,400),(0,255,0),0)
            crop_image = cv_image[170:400, 170:400]
            #crop_image = cv_image
            height, width, channels = crop_image.shape

            blur = cv2.blur(crop_image,(7,7), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            lower_range = np.array([2,0,0])
            upper_range = np.array([10,255,255])

            mask = cv2.inRange(hsv,lower_range,upper_range)
            skinkernel = np.ones((5,5))

            dilation = cv2.dilate(mask, skinkernel, iterations = 2)
            erosion = cv2.erode(dilation, skinkernel, iterations = 1)

            filtered = cv2.GaussianBlur(dilation, (15,15), 1)
            ret,thresh = cv2.threshold(filtered, 127, 255, 0)

            Label_ret, markers = cv2.connectedComponents(thresh)
            num = markers.max()
            #print(num)

            for i in range(1, num+1):
                pts =  np.where(markers == i)
                if len(pts[0]) < 100:
                    markers[pts] = 0

            label_hue = np.uint8(markers.copy())
            label_hue = np.where(label_hue != 0, 255, label_hue)

            res = cv2.bitwise_and(crop_image,crop_image,mask = label_hue)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)


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

                self.boolFlag.data = 0

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
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if l == 4:
                        cv2.putText(cv_image,"Go Forward", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        
                        self.boolFlag.data = 1
            
                    #show the windows
                    #cv2.imshow('hand_binary',label_hue)
                #cv2.imshow('frame',cv_image)
                #cv2.imshow('hand',res)
            
            
            # Get the good feature keypoints in the selected region
            #keypoints = self.get_keypoints(crop_image, self.detect_box)
            """            
            # If we have points, display them
            if keypoints is not None and len(keypoints) > 0:
                self.marker_image = np.zeros_like(cv_image)
                xs,ys,w,h = self.selection
                cv2.rectangle(self.marker_image, (xs, ys), (xs + w, ys + h), (0, 255, 255), 1)
                for x, y in keypoints:
                    cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), 0, 8, 0)
            
            
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
            """

            
        except Exception as e:
            print e
            
                                
        return cv_image

    def get_keypoints(self, input_image, detect_box):
        # Initialize the mask with all black pixels
        self.mask = np.zeros_like(input_image)
 
        # Get the coordinates and dimensions of the detect_box
        try:
            x, y, w, h = detect_box
        except: 
            return None
        
        # Set the selected rectangle within the mask to white
        self.mask[y:y+h, x:x+w] = 255

        # Compute the good feature keypoints within the selected region
        keypoints = list()
        kp = cv2.goodFeaturesToTrack(input_image, mask = self.mask, **self.gf_params)
        if kp is not None and len(kp) > 0:
            for x, y in np.float32(kp).reshape(-1, 2):
                keypoints.append((x, y))
                
        return keypoints

if __name__ == '__main__':
    try:
        node_name = "good_features"
        goodfeatures= GoodFeatures(node_name)
        #hand_flag = None
        #pub = rospy.Publisher('hand_detect_forward', Bool, queue_size=10)
        
        #rate = rospy.Rate(10) # 10hz
        
        while not rospy.is_shutdown():
            if goodfeatures.display_image is not None:
                goodfeatures.image_show(goodfeatures.cv_window_name, goodfeatures.display_image)

            #pub.publish(self.hand_flag)
            #f = Bool()
            #f.data = hand_flag
            #print(hand_flag)
            hand_pub = rospy.Publisher("/gesture_forward", Bool, queue_size=1)
            hand_pub.publish(goodfeatures.boolFlag)
                
    except KeyboardInterrupt:
        print "Shutting down the Good Features node."
        cv.DestroyAllWindows()

