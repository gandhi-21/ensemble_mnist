import numpy as np
import pandas as pd
import tensorflow as tf

from helper import conv2d, max_pool_2x2, dropout_layer, conv2d_layers, max_pool_layers

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Defining the constants
LEARNING_RATE = 1e-4
EPOCHS = 5000
BATCH_SIZE = 128
image_size = 28
labels_size = 10
DROPOUT = 0.5

# Define the placeholder variables for the graph
X = tf.placeholder(tf.float32, shape=[None, image_size*image_size])
y = tf.placeholder(tf.float32, shape=[None, labels_size])
keep_prob = tf.placeholder(tf.float32)

def model2(x, weights, biases, dropout):
    # # Make the tensorflow graph here

    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # # conv1 conv2 and max1 and drop1
    # conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    # conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    # conv2 = max_pool_2x2(conv2, ksize=2)
    # conv2 = dropout_layer(conv2, keep_prob=dropout)

    # # con3 conv4 and max2 and drop2
    # conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])
    # conv4 = conv2d(conv3, weights["wc4"], biases["bc4"])
    # conv4 = max_pool_2x2(conv4, ksize=2)
    # conv4 = dropout_layer(conv4, keep_prob=dropout)
    
    # # conv5, conv6, max3 and drop3
    # conv5 = conv2d(conv4, weights["wc5"], biases["bc5"])
    # conv6 = conv2d(conv5, weights["wc6"], biases["bc6"])
    # conv6 = max_pool_2x2(conv6, ksize=2)
    # conv6 = dropout_layer(conv6, keep_prob=dropout)

    # conv6 = tf.layers.flatten(conv6)

    # # flatten, dense 1 and dropout
    # # fc1 = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(conv6, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, keep_prob=dropout)

    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d_layers(x, filters=32, kernel_size=(5, 5))
    conv2 = conv2d_layers(conv1, filters=32, kernel_size=(5, 5))
    conv2 = max_pool_layers(conv2)

    conv2 = tf.layers.flatten(conv2)

    fc1 = tf.reshape(conv2, )

    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'wc5': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'wc6': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'wd1': tf.Variable(tf.random_normal([1024, 256])),
    'out': tf.Variable(tf.random_normal([256, labels_size]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),
    'bc6': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([labels_size]))
}

# Make the model variable
logits = model2(X, weights, biases, keep_prob)
predictions = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

# Evaluate the model
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variable
init = tf.global_variables_initializer()

display_step = 100

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

# Start Training
with tf.Session(config=config) as sess:
    
    sess.run(init)

    for step in range(1, EPOCHS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        # Run optimization op
        sess.run(train_op, feed_dict={X: batch_x, y: batch_y, keep_prob: 0.7})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
            print(f"Step {step}, Minibatch loss: {loss}, Training accuracy: {acc}")

    print("Optimization finished")

    print("Testing accuracy: ", \
        sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
