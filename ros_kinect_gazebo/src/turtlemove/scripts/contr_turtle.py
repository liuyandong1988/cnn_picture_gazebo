#!/usr/bin/env python
# coding=utf-8

import rospy
from geometry_msgs.msg import Twist


def contr(keynumber):
    # turtlesim使用topic
    pub = rospy.Publisher('~cmd_vel', Twist, queue_size=5)
    countnum = 0
    if keynumber == 3:
        while(1):
            twist = Twist()
            twist.linear.x = 0.2
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0.14
            pub.publish(twist)
            countnum += 1

            if countnum > 100000:
                countnum = 0
                exit(0)
