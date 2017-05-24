cmake_minimum_required(VERSION 2.8.3)
project(capture)

## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  std_msgs
  image_transport
)

## Declare a catkin package  
catkin_package()
    
## Build 
include_directories(include ${catkin_INCLUDE_DIRS})  

## Declare a C++ executable

add_executable(kinectcapture src/open_rec_contours.cpp)
add_executable(capicture src/capicture.cpp)
#depend lib
 target_link_libraries(kinectcapture
   ${catkin_LIBRARIES}
 )

 target_link_libraries(capicture
   ${catkin_LIBRARIES}
 )


