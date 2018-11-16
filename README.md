# AlexaBot

--------------------------
To initialize turtlebot: `roslaunch turtlebot_bringup minimal.launch`


To start kinect on turtlebot: `roslaunch freenect_launch freenect.launch`


To start alexa-control node: `roslaunch teleop_twist_keyboard voice_turtlebot.launch`


To launch the usb-camera on host: `roslaunch usb_cam usb_cam-test.launch`


To launch the hand_gesture detection: `./good_hand_CNN.py`


To launch json2rosmessage: `./jsonpub.py`

In latest version, `roslaunch teleop_twist_keyboard voice_turtlebot.launch` has integrated `./jsonpub.py` command


To launch turtlebot follwer `roslaunch turtlebot_follower follower.launch`


To check the turtlbot follwer simulation `roslaunch turtlebot_follower simulate_follower.launch`


To enable following function, `rostopic pub -1 /Alexa/follow_cmd 1`


--------------------------
Face detection : branch Jinxin_CV_facedetection 


for iris detection just run 'eyeball and iris tracking ' part, you dont need to make any change. 



Pay attention!!!! All the opencv code is run in OPENCV 3.0


# To do List
1. voice_control.py should publish once, when it receives a series of data; hand_detect_CNN.py and voice_control.py Improved!
