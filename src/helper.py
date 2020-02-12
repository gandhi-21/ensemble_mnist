import numpy as np
import pandas as pd
import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool_2x2(x, ksize=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')

def dropout_layer(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)

def dense_to_onehot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot