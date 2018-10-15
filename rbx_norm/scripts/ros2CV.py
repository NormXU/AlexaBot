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
		rospy.on_shutdown(self.cleanup)

		self.frame = None
		self.frame_width = None
		self.frame_height = None
		self.frame_size = None
		self.keystroke = None


		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)


	def image_callback(self,data):
		# Store the image header in a global variable
        self.image_header = data.header

		frame = self.convert_image(data)

		if self.frame_width is None:
			self.frame_size = (frame.shape[1], frame.shape[0])
			self.frame_width, self.frame_height = self.frame_size

		self.frame = frame.copy()
		processed_image = self.processing(frame)
		self.processed_image = processed_image.copy()


	def convert_image(self, ros_image):
		try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")       
            return np.array(cv_image, dtype=np.uint8)
        except CvBridgeError, e:
            print e

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



