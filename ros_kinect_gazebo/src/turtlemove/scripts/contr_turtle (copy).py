#!/usr/bin/env python
#coding=utf-8
import rospy
from std_msgs.msg import String
import time


def contr( keynumber ):
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('number', String, queue_size=10)


    rate = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        num = str('%d'%keynumber)
        pub.publish(num)
        rospy.loginfo(num)
        rate.sleep()

    # num = str('%d'%keynumber)
    # pub.publish(num)
    # rospy.loginfo(num)
