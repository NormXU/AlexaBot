# AlexaBot

--------------------------
To initialize turtlebot: `roslaunch turtlebot_bringup minimal.launch`


To start kinect on turtlebot: `roslaunch freenect_launch freenect.launch`


To start alexa-control node: `roslaunch teleop_twist_keyboard voice_turtlebot.launch`


To launch the usb-camera on host: `roslaunch usb_cam usb_cam-test.launch`


To launch the hand_gesture detection: `roslaunch rbx_norm hand_detect.launch`


To launch json2rosmessage: `./transform_json.py xxx.json`


To launch turtlebot follwer `roslaunch turtlebot_follower follower.launch`

To check the turtlbot follwer simulation `roslaunch turtlebot_follower simulate_follower.launch`



--------------------------
Face detection : branch Jinxin_CV_facedetection 


for iris detection just run 'eyeball and iris tracking ' part, you dont need to make any change. 



Pay attention!!!! All the opencv code is run in OPENCV 3.0
