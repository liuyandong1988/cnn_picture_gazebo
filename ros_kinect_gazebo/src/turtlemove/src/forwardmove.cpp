/*
 * try.cpp
 *
 *  Created on: Apr 6, 2017
 *      Author: exbot
 */

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

int main(int argc, char **argv) {

  ros::init(argc, argv, "forwardmove");
  ros::NodeHandle n;

  ros::Publisher vel_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 10);

  ros::Rate loop_rate(10);

  while (ros::ok()) {
    geometry_msgs::Twist msg_vel;

    msg_vel.linear.x = 0.2;
    msg_vel.linear.y = msg_vel.linear.z = 0;

    msg_vel.angular.x = msg_vel.angular.y = msg_vel.angular.z = 0;

    vel_pub.publish(msg_vel);
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

