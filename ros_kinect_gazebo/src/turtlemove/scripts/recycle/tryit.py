#!/usr/bin/env python
# license removed for brevity
import rospy
import os
import tensorflow as tf
import model
import time
from datetime import datetime
from data_input import encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch
from numpy.distutils.fcompiler import none


def talker():

    rospy.init_node('talker', anonymous=True)
    if os.path.exists('/home/exbot/tensor/my_cnn_real/src/one_pictest/g1.png') == True:
	rospy.loginfo("OK!! ")
    else:
        rospy.loginfo("wuwu ")

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
