#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
from datetime import datetime

import numpy as np
# from six.moves import xrange
import tensorflow as tf


# 用 get_variable 在 CPU 上定义常量
def variable_on_cpu(name, shape, initializer = tf.constant_initializer(0.1)):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, 
                              dtype = dtype)
    return var

 # 用 get_variable 在 CPU 上定义变量
def variables(name, shape, stddev): 
    dtype = tf.float32
    var = variable_on_cpu(name, shape, 
                          tf.truncated_normal_initializer(stddev = stddev, 
                                                          dtype = dtype))
    return var
    
# 定义网络结构
def inference(images, BATCH_SIZE, NUM_CLASSES):
    """Build the CIFAR-10 model.

      Args:
        images: Images returned from distorted_inputs() or inputs().
        BATCH_SIZE:64 image numbers

      Returns:
        Logits.
    """
    with tf.variable_scope('conv1') as scope:
        # 用 5*5 的卷积核，64 个 Feature maps
        weights = variables('weights', [5,5,3,64], 5e-2)
        # 卷积，步长为 1*1
        conv = tf.nn.conv2d(images, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # 通过 ReLu 激活函数
        conv1 = tf.nn.relu(bias, name = scope.name)
        # 柱状图总结 conv1
        tf.summary.histogram(scope.name + '/activations', conv1)
        
    with tf.variable_scope('pooling1_lrn') as scope:
        # 最大池化，3*3 的卷积核，2*2 的卷积
        pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')
        # 局部响应归一化
        norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')

    with tf.variable_scope('conv2') as scope:
        weights = variables('weights', [5,5,64,64], 5e-2)
        conv = tf.nn.conv2d(norm1, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope.name)
        tf.summary.histogram(scope.name + '/activations', conv2)
        
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')        
        pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')

    with tf.variable_scope('fully_con1') as scope:
        # 第一层全连接
        reshape = tf.reshape(pool2, [BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        weights = variables('weights', shape=[dim,384], stddev=0.004)
        biases = variable_on_cpu('biases', [384])
        # ReLu 激活函数
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, 
                            name = scope.name)
        # 柱状图总结 local3
        tf.summary.histogram(scope.name + '/activations', local3)
        
    with tf.variable_scope('fully_con2') as scope:
        # 第二层全连接
        weights = variables('weights', shape=[384,192], stddev=0.004)
        biases = variable_on_cpu('biases', [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, 
                            name = scope.name)
        tf.summary.histogram(scope.name + '/activations', local4)
        
    with tf.variable_scope('softmax_linear') as scope:
        # softmax 层，实际上不是严格的 softmax ，真正的 softmax 在损失层
        weights = variables('weights', [192, NUM_CLASSES], stddev=1/192.0)
        biases = variable_on_cpu('biases', [NUM_CLASSES])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, 
                                name = scope.name)
        tf.summary.histogram(scope.name + '/activations', softmax_linear)
        
    return softmax_linear

# 交叉熵损失层             

def loss(logits, labels):
     with tf.variable_scope('loss') as scope:
#         labels = tf.cast(labels, tf.int64)
        # 交叉熵损失，至于为什么是这个函数，后面会说明。
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                             (logits, labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name + '/x_entropy', loss)
        return loss