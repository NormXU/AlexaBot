#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
previous_FWD = None
previous_RGT = None
previous_LFT = None
#previous_timestp = None
cnt = 0
def Fwdcallback(data):
	global previous_FWD
	#global previous_timestp
	global cnt
	if previous_FWD is None:
		previous_FWD = data.data
		#now = rospy.get_rostime()
		#previous_timestp = now
		#print(previous)
	else:
		if data.data != previous_FWD:
			#print(data.data)
			fwdPub.publish(data)
			if previous_FWD == True:
				cnt = cnt + 1
			previous_FWD = data.data

def Rightcallback(data):
	global previous_RGT
	#global previous_timestp
	global cnt
	if previous_RGT is None:
		previous_RGT = data.data
		#now = rospy.get_rostime()
		#previous_timestp = now
		#print(previous)
	else:
		if data.data != previous_RGT:
			#print(data.data)
			rightPub.publish(data)
			if previous_RGT == True:
				cnt = cnt + 1
			previous_RGT = data.data
			
def Leftcallback(data):
	global previous_LFT
	#global previous_timestp
	global cnt
	if previous_LFT is None:
		previous_LFT = data.data
		#now = rospy.get_rostime()
		#previous_timestp = now
		#print(previous)
	else:
		if data.data != previous_LFT:
			#print(data.data)
			leftPub.publish(data)
			if previous_LFT == True:
				cnt = cnt + 1
			previous_LFT = data.data

    
def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/Alexa/fwd_cmd_realtime", Bool, Fwdcallback)
    rospy.Subscriber("/Alexa/right_cmd_realtime", Bool, Rightcallback)
    rospy.Subscriber("/Alexa/left_cmd_realtime", Bool, Leftcallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
	fwdPub = rospy.Publisher('Alexa/forward_cmd', Bool, queue_size=10)
	rightPub = rospy.Publisher('Alexa/right_cmd', Bool, queue_size=10)
	leftPub = rospy.Publisher('Alexa/left_cmd', Bool, queue_size=10)
	listener()