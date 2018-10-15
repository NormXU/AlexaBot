#!/usr/bin/env python

import rospy
import cv2
import cv2 as cv
import sys
from rbx1_vision.ros2opencv2 import ROS2OpenCV2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np

class CamShitfNode(ROS2OpenCV2):
	def __init__(self, node_name):
		ROS2OpenCV2.__init__(self, node_name)

		self.node_name = node_name

		self.smin = rospy.get_param("~smin", 85)
		self.vmin = rospy.get_param("~vmin", 85)
		self.vmax = rospy.get_param("~vmax", 85)
		self.threshold = rospy.get_param("~threshold", 50)
		

		cv2.namedWindow("Histogram", cv2.WINDOW_NORMAL)
		cv2.moveWindow("Histogram", 700, 50)
		cv2.namedWindow("Parameters", 0)
		cv2.moveWindow("Parameters", 700, 325)
		cv2.namedWindow("Backproject", 0)
		cv2.moveWindow("Backproject", 700, 600)

		# Create Track Bar
		cv2.createTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
		cv2.createTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
		cv2.createTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)
		cv2.createTrackbar("Threshold", "Parameters", self.threshold, 255, self.set_threshold)

		#Variables
		self.hist = None
		self.track_window = None
		self.show_backproj = False
		self.imhis = None
		self.back  = None

	def processing(self, cv_image):
		# print('Y')
		
		frame = cv2.blur(cv_image, (5, 5))
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
		
		if self.selection is not None:
			x0, y0, w, h = self.selection
			x1 = x0 + w
			y1 = y0 + h
			self.track_window = (x0, y0, x1, y1)
			hsv_roi = hsv[y0:y1, x0:x1]
			mask_roi = mask[y0:y1, x0:x1]
			self.hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
			cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX);
			self.hist = self.hist.reshape(-1)
			self.show_hist()

		
		if self.hist is not None:
			backproject = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
			backproject &= mask
			ret, backproject = cv2.threshold(backproject, self.threshold, 255, cv.THRESH_TOZERO)
			x, y, w, h = self.track_window

			if self.track_window is None or w <= 0 or h <=0:
				self.track_window = 0, 0, self.frame_width - 1, self.frame_height - 1

			# Set the criteria for the CamShift algorithm
			term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
			self.track_box, self.track_window = cv2.CamShift(backproject, self.track_window, term_crit)

			self.back = backproject
			#print(backproject.shape)


		return cv_image

      
	def show_hist(self):
		#print('Y')
		bin_w = 24
		img = np.zeros((256, self.hist.shape[0]*bin_w, 3), np.uint8)
		bin_count = self.hist.shape[0]
		for i in range(bin_count):
			h = int(self.hist[i])
			cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
		img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
		self.imhis = img
		#print(img.shape)
		#cv2.imshow('Histogram', img)
		
		


	# Callback Functions

	def set_smin(self, pos):
		self.smin = pos

	def set_vmin(self, pos):
		self.vmin = pos

	def set_vmax(self, pos):
		self.vmax = pos

	def set_threshold(self, pos):
		self.threshold = pos




def main(args):
	node_name = "camshift"
	Ctmp = CamShitfNode(node_name)
	while not rospy.is_shutdown():
		if Ctmp.display_image is not None:
			Ctmp.image_show(Ctmp.cv_window_name, Ctmp.display_image)
			if Ctmp.imhis is not None:
				cv2.imshow('Histogram', Ctmp.imhis)
				cv2.imshow("Backproject", Ctmp.back)

		cv2.waitKey(5)
			#cv2.namedWindow("Histogram", cv2.WINDOW_NORMAL)
			#cv2.createTrackbar("Saturation", "Histogram", 0, 255, ros2opencv.nothing)
	cv2.destroyAllWindows()
	rospy.spin()
        
 
if __name__ == '__main__':
	main(sys.argv)
