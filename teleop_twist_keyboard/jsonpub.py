#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
import requests
import json
import time

if __name__ == '__main__':
	pub = rospy.Publisher('json_data', String, queue_size = 10)
	rospy.init_node('path_publisher', anonymous = True)
	fwdPub = rospy.Publisher('Alexa/forward_cmd', Bool, queue_size = 10)
	lftPub = rospy.Publisher('Alexa/left_cmd', Bool, queue_size = 10)
	rgtPub = rospy.Publisher('Alexa/right_cmd', Bool, queue_size = 10)
	
	while 1:
		r =  requests.get('http://hopai.club/get-commands')
		parsedResponse = r.json()
		#print("Here")
		for indResponse in parsedResponse:
			data={}
			data['turtlebot_enable'] = False
			data['fwd_enable'] = False
			data['follow_enable'] = False
			data['right_enable'] = False
			data['left_enable'] = False

			if indResponse['command'] == 'forward':
				data['fwd_enable'] = True
				fwdPub.publish(data['fwd_enable'])
			elif indResponse['command'] == 'backward':
				data['fwd_enable'] = True
				fwdPub.publish(data['fwd_enable'])
			elif indResponse['command'] == 'left':
				data['left_enable'] = True
				lftPub.publish(data['left_enable'])
			elif indResponse['command'] == 'right':
				data['right_enable'] = True
				rgtPub.publish(data['right_enable'])


		#with open('/home/norm/catkin_ws/src/ur10_planner/scripts/waypoints.json','r') as myfile:
		#	data = myfile.read()
			#print(r.json())	
			pub.publish(json.dumps(data))

			rospy.loginfo("json data publishing ends")
		time.sleep(0.5)

    
