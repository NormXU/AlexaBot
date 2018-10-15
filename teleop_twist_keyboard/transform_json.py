#!/usr/bin/env python
import os,sys
from std_msgs.msg import String
from std_msgs.msg import Bool
import rospy
from rospy_message_converter import json_message_converter
from teleop_twist_keyboard.msg import Alexabot

if __name__ == '__main__':
    #json_str = '{"data1": true, "data2":true}'
    with open( sys.argv[1], 'r') as myfile:
        json_str=myfile.read()
    message = json_message_converter.convert_json_to_ros_message('teleop_twist_keyboard/Alexabot', json_str)
    pub = rospy.Publisher('my_topic', Alexabot, queue_size=10)
    rospy.init_node('path_publisher', anonymous=True)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
    	pub.publish(message)
    	rate.sleep()
    

    #with open( sys.argv[1], 'r') as myfile:
        #data=myfile.read()
        #pub.publish(data)
