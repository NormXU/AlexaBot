#!/usr/bin/env python

from __future__ import print_function

import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import sys, select, termios, tty
import time


msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""
x = 0
y = 0
z = 0
th = 0
status = 0

moveBindings = {
		'i':(1,0,0,0),
		'o':(1,0,0,-1),
		'j':(0,0,0,1),
		'l':(0,0,0,-1),
		'u':(1,0,0,1),
		',':(-1,0,0,0),
		'.':(-1,0,0,1),
		'm':(-1,0,0,-1),
		'O':(1,-1,0,0),
		'I':(1,0,0,0),
		'J':(0,1,0,0),
		'L':(0,-1,0,0),
		'U':(1,1,0,0),
		'<':(-1,0,0,0),
		'>':(-1,-1,0,0),
		'M':(-1,1,0,0),
		't':(0,0,1,0),
		'b':(0,0,-1,0),
	       }

speedBindings={
		'q':(1.1,1.1),
		'z':(.9,.9),
		'w':(1.1,1),
		'x':(.9,1),
		'e':(1,1.1),
		'c':(1,.9),
	      }


Total_Times = 20;

def fwdCb(msg):
	if msg.data is True:
		print('Go Forward')
		count_times = 0;
		while (count_times < Total_Times):
			key = 'i'
			x = moveBindings[key][0]
			y = moveBindings[key][1]
			z = moveBindings[key][2]
			th = moveBindings[key][3]
			twistfwd = Twist()
			twistfwd.linear.x = x*speed; twistfwd.linear.y = y*speed; twistfwd.linear.z = z*speed;
			twistfwd.angular.x = 0; twistfwd.angular.y = 0; twistfwd.angular.z = th*turn
			pub.publish(twistfwd)
			time.sleep(0.2)
			count_times+=1
		print('Command Finish')

def leftCb(msg):
	if msg.data is True:
		print('Turn left')
		count_times = 0;
		while (count_times < Total_Times):
			key = 'j'
			x = moveBindings[key][0]
			y = moveBindings[key][1]
			z = moveBindings[key][2]
			th = moveBindings[key][3]
			twistleft = Twist()
			twistleft.linear.x = x*speed; twistleft.linear.y = y*speed; twistleft.linear.z = z*speed;
			twistleft.angular.x = 0; twistleft.angular.y = 0; twistleft.angular.z = th*turn
			pub.publish(twistleft)
			time.sleep(0.2)
			count_times+=1
		print('Command Finish')

def rightCb(msg):
	if msg.data is True:
		print('Turn Right')
		count_times = 0;
		while (count_times < Total_Times):
			key = 'l'
			x = moveBindings[key][0]
			y = moveBindings[key][1]
			z = moveBindings[key][2]
			th = moveBindings[key][3]
			twistright = Twist()
			twistright.linear.x = x*speed; twistright.linear.y = y*speed; twistright.linear.z = z*speed;
			twistright.angular.x = 0; twistright.angular.y = 0; twistright.angular.z = th*turn
			pub.publish(twistright)
			time.sleep(0.2)
			count_times+=1
		print('Command Finish')

def StopCb(msg):
	if msg.data is True:
		print('Stop')

		x = 0
		y = 0
		z = 0
		th = 0
		twist = Twist()
		twist.linear.x = x*speed; twist.linear.y = y*speed; twist.linear.z = z*speed;
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th*turn
		pub.publish(twist)
		time.sleep(0.1)
		print('Command Finish')

def getKey():
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key


def vels(speed,turn):
	return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    	settings = termios.tcgetattr(sys.stdin)
	
	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
	rospy.init_node('teleop_voice')

	rospy.Subscriber("forward_cmd", Bool, fwdCb)
	rospy.Subscriber("left_cmd", Bool, leftCb)
	rospy.Subscriber("right_cmd", Bool, rightCb)
	rospy.Subscriber("stop_cmd", Bool, StopCb)


	speed = rospy.get_param("~speed", 0.5)
	turn = rospy.get_param("~turn", 1.0)


	try:
		#print(msg)
		print(vels(speed,turn))

		while(1):
			key = getKey()
			if key in moveBindings.keys():
				x = moveBindings[key][0]
				y = moveBindings[key][1]
				z = moveBindings[key][2]
				th = moveBindings[key][3]
			elif key in speedBindings.keys():
				speed = speed * speedBindings[key][0]
				turn = turn * speedBindings[key][1]

				print(vels(speed,turn))
				if (status == 14):
					print(msg)
				status = (status + 1) % 15
			else:
				x = 0
				y = 0
				z = 0
				th = 0
				if (key == '\x03'):
					break

			twist = Twist()
			twist.linear.x = x*speed; twist.linear.y = y*speed; twist.linear.z = z*speed;
			twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th*turn
			pub.publish(twist)

	except Exception as e:
		print(e)

	finally:
		twist = Twist()
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		#pub.publish(twist)

    		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


