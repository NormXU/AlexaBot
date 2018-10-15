#!/usr/bin/env python


import rospy
import cv2
import cv2 as cv
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np

class ROS2OpenCV2(object):
    def __init__(self, node_name, *args, **kwargs):        
        self.node_name = node_name

        rospy.init_node(node_name)
        rospy.loginfo("Starting node " + str(node_name))

        rospy.on_shutdown(self.cleanup)
        
        # A number of parameters to determine what gets displayed on the
        # screen. These can be overridden the appropriate launch file
        self.show_text = rospy.get_param("~show_text", True)
        self.show_features = rospy.get_param("~show_features", True)
        self.show_boxes = rospy.get_param("~show_boxes", True)
        self.flip_image = rospy.get_param("~flip_image", False)
        self.feature_size = rospy.get_param("~feature_size", 1)

        # Initialize the Region of Interest and its publisher
        self.ROI = RegionOfInterest()
        self.roi_pub = rospy.Publisher("/roi", RegionOfInterest, queue_size=1)
        
        # Initialize a number of global variables
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.depth_image = None
        self.marker_image = None
        self.display_image = None
        self.grey = None
        self.prev_grey = None
        self.selected_point = None
        self.selection = None
        self.drag_start = None
        self.keystroke = None
        self.detect_box = None
        self.track_box = None
        self.display_box = None
        self.keep_marker_history = False
        self.night_mode = False
        self.auto_face_tracking = False
        self.cps = 0 # Cycles per second = number of processing loops per second.
        self.cps_values = list()
        self.cps_n_values = 20
        self.busy = False
        self.resize_window_width = 0
        self.resize_window_height = 0
        self.face_tracking = False
        self.show_image = True
        
        # Create the main display window
        if self.show_image:
            self.cv_window_name = self.node_name
            cv2.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)
            if self.resize_window_height > 0 and self.resize_window_width > 0:
                cv.ResizeWindow(self.cv_window_name, self.resize_window_width, self.resize_window_height)

            # Set a call back on mouse clicks on the image window
            #cv2.setMouseCallback (self.node_name, self.on_mouse_click, None)

        # create the cv_bridge object
        self.bridge = CvBridge()
        
        # Subscribe to the image and depth topics and set the appropriate callbacks
        # The image topic names can be remapped in the appropriate launch file
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        #self.depth_sub = rospy.Subscriber("/kinect2/sd/image_depth", Image, self.depth_callback, queue_size=1)
                                    
            
    def image_callback(self, data):
        # Store the image header in a global variable
        self.image_header = data.header

        # Time this loop to get cycles per second
        start = time.time()
        
        # Convert the ROS image to OpenCV format using a cv_bridge helper function
        frame = self.convert_image(data)
        
        # Some webcams invert the image
        if self.flip_image:
            frame = cv2.flip(frame, 0)
        
        # Store the frame width and height in a pair of global variables
        if self.frame_width is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size            
            
        # Create the marker image we will use for display purposes
        if self.marker_image is None:
            self.marker_image = np.zeros_like(frame)
            
        # Copy the current frame to the global image in case we need it elsewhere
        self.frame = frame.copy()

        # Reset the marker image if we're not displaying the history
        if not self.keep_marker_history:
            self.marker_image = np.zeros_like(self.marker_image)
        
        # Process the image to detect and track objects or features
        processed_image = self.processing(frame)
        # print('Y')
        
        # If the result is a greyscale image, convert to 3-channel for display purposes """
        #if processed_image.channels == 1:
            #cv.CvtColor(processed_image, self.processed_image, cv.CV_GRAY2BGR)
        #else:
        
        # Make a global copy
        self.processed_image = processed_image.copy()
        
        # Display the user-selection rectangle or point 
        # self.display_selection()
        
        # Night mode: only display the markers
        if self.night_mode:
            self.processed_image = np.zeros_like(self.processed_image)
            
        # Merge the processed image and the marker image
        self.display_image = cv2.bitwise_or(self.processed_image, self.marker_image)

        # If we have a track box, then display it.  The track box can be either a regular
        # cvRect (x,y,w,h) or a rotated Rect (center, size, angle).
        
        # Publish the ROI
        #self.publish_roi()
            
        # Compute the time for this loop and estimate CPS as a running average
               
    def image_show(self, window_name, display_image):
        # Update the image display
        if self.show_image:
            cv2.imshow(window_name, display_image)
            
            self.keystroke = cv2.waitKey(5)
        
            # Process any keyboard commands
            if self.keystroke is not None and self.keystroke != -1:
                try:
                    cc = chr(self.keystroke & 255).lower()
                    if cc == 'n':
                        self.night_mode = not self.night_mode
                    elif cc == 'f':
                        self.show_features = not self.show_features
                    elif cc == 'b':
                        self.show_boxes = not self.show_boxes
                    elif cc == 't':
                        self.show_text = not self.show_text
                    elif cc == 'q':
                        # The has press the q key, so exit
                        rospy.signal_shutdown("User hit q key to quit.")
                except:
                    pass
                
          
    def convert_image(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")       
            return np.array(cv_image, dtype=np.uint8)
        except CvBridgeError, e:
            print e
          
    
            
    
          
    def processing(self, frame): 
        return frame
    
    
        
        
    def cleanup(self):
        print "Shutting down vision node."
        if self.show_image:
            cv2.destroyAllWindows()       

def main(args):    
    try:
        node_name = "ros2opencv2"
        ros2opencv = ROS2OpenCV2(node_name)
        while not rospy.is_shutdown():
            if ros2opencv.display_image is not None:
                ros2opencv.image_show(ros2opencv.cv_window_name, ros2opencv.display_image)

    except KeyboardInterrupt:
        print "Shutting down ros2opencv node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
