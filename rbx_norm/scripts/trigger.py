#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
previous = None
previous_timestp = None
cnt = 0
def callback(data):
	global previous
	global previous_timestp
	global cnt
	if previous is None:
		previous = data.data
		now = rospy.get_rostime()
		previous_timestp = now
		#print(previous)
	else:
		if data.data != previous:
			#print(data.data)
			fwdPub.publish(data)
			if previous == True:
				cnt = cnt + 1
			previous = data.data


    
def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/Alexa/fwd_cmd_realtime", Bool, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
	fwdPub = rospy.Publisher('Alexa/trigger_fwd', Bool, queue_size=10)
	listener()