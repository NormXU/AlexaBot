#!/usr/bin/env python

import sys
import rospy
from turtlebot_msgs.srv import *

from std_msgs.msg import Bool



#x = int(sys.argv[1])
x = 0

def followCb(msg):
	if msg.data is True:
		rospy.loginfo('Start')
		x = 1
		follow_client(x)
	else:
		rospy.loginfo('Stop')
		x = 0
		follow_client(x)



def follow_client(x):
    rospy.wait_for_service('/turtlebot_follower/change_state')
    try:
        follow_state = rospy.ServiceProxy('/turtlebot_follower/change_state', SetFollowState)
        resp1 = follow_state(x)
        return resp1.result
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":

	rospy.init_node('follow_listener', anonymous=True)
	rospy.Subscriber("follow_cmd", Bool, followCb)

	
	rospy.spin()