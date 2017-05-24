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
from std_msgs.msg import String
import contr_turtle

BATCH_SIZE = 1

LEARNING_RATE = 0.1
MAX_STEP = 10000
NUM_CLASSES = 4


train_dir_mydata = '/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/log_good/'
recognize_label = {0: 'chair', 1: 'door', 2: 'garbage', 3: 'bookself'}


def evaluate():
     # compare with labels fetch accuracy
    encode_to_tfrecords("/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/one_pictest/test.txt",
                        "/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/one_pictest", 'test.tfrecords', (37, 37))
    test_image, test_label = decode_from_tfrecords(
        '/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/one_pictest/test.tfrecords', num_epoch=None)
    test_images, test_labels = get_test_batch(
        test_image, test_label, batch_size=BATCH_SIZE, crop_size=32)
    # [batch, in_height, in_width, in_channels]
    test_images = tf.reshape(test_images, shape=[-1, 32, 32, 3])
    test_images = (tf.cast(test_images, tf.float32) / 255. - 0.5) * 2  # guiyi

    logits = model.inference(test_images, BATCH_SIZE, NUM_CLASSES)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        rospy.loginfo("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_dir_mydata)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
            saver.restore(sess, os.path.join(train_dir_mydata, ckpt_name))
            rospy.loginfo('Loading success, global_step is %s' % global_step)
            test_number = sess.run(tf.argmax(logits, 1))
            rospy.loginfo('*****recognized label:%s' %
                          recognize_label[test_number[0]] + '*****')
            os.remove(
                "/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/one_pictest/test.png")
            contr_turtle.contr(test_number[0])

        coord.request_stop()  # queue close
        coord.join(threads)


def identify():

    rospy.init_node('turtle')
    if os.path.exists('/home/exbot/ros_kinect_gazebo/src/turtlemove/scripts/one_pictest/test.png'):
        evaluate()
    else:
        exit()


if __name__ == '__main__':
    try:
        identify()
    except rospy.ROSInterruptException:
        pass
