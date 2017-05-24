cmake_minimum_required(VERSION 2.8.3)
project(turtlemove)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
)


## Declare a catkin package  
catkin_package()
    
## Build 
include_directories(include ${catkin_INCLUDE_DIRS})  

## Declare a C++ executable
 add_executable(forwardmove src/forwardmove.cpp)

#depend lib
 target_link_libraries(forwardmove
   ${catkin_LIBRARIES}
 )
