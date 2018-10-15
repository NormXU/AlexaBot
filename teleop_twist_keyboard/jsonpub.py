#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import String

if __name__ == '__main__':
	pub = rospy.Publisher('json_data', String, queue_size = 10)
	rospy.init_node('path_publisher', anonymous = True)
	with open('/home/norm/catkin_ws/src/ur10_planner/scripts/waypoints.json','r') as myfile:
		data = myfile.read()
		
		while not rospy.is_shutdown():
			pub.publish(data)
			
        rospy.loginfo("json data publishing ends")
    
