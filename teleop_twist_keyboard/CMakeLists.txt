cmake_minimum_required(VERSION 2.8.3)
project(teleop_twist_keyboard)

find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
)

add_message_files(
  FILES
  Alexabot.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  
)

catkin_package(
CATKIN_DEPENDS message_runtime
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   teleop_twist_keyboard_cmd.py
   teleop_voice.py
   stateChangeVoice.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

