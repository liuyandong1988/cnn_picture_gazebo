#!/usr/bin/env python
# license removed for brevity
import rospy
import os
import tensorflow as tf
import model
import time
from datetime import datetime
from data_input import encode_to_tfrecords, decode_from_tfrecords, get_batch, get_test_batch
from numpy.distutils.fcompiler import none


def evaluate():
    rospy.loginfo("haha!! ")


def identify():

    rospy.init_node('identify', anonymous=True)
    if os.path.exists('/home/exbot/tensor/my_cnn_real/src/one_pictest/g1.png') == True:
        evaluate()


if __name__ == '__main__':
    try:
        identify()
    except rospy.ROSInterruptException:
        pass
