import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from helper import conv2d, max_pool_2x2, dense_to_onehot


# train_df = pd.read_csv('../input/train.csv')
# y_train = train_df['label']
# x_train = train_df.drop(labels = ['label'], axis=1)

# x_train = x_train / 255.0
# train_x, x_test, train_y, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# train_y = dense_to_onehot(train_y, 10)
# train_y = train_y.astype(np.uint8)

# y_test = dense_to_onehot(y_test, 10)
# y_test = y_test.astype(np.uint8)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define the constants
LEARNING_RATE = 1e-4
EPOCHS = 2000
BATCH_SIZE = 128
image_size = 28
labels_size = 10
DROPOUT = 0.8


# Define the placeholder variables for the graph
X = tf.placeholder(tf.float32, shape=[None, image_size*image_size], name="x")
y = tf.placeholder(tf.float32, shape=[None, labels_size])
keep_prob = tf.placeholder(tf.float32)


# Make the tensorflow graph
def model1(x, weights, biases, dropout):

    with tf.name_scope('reshape'):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        conv1 = conv2d(x, weights["wc1"], biases['bc1'])
        conv1 = max_pool_2x2(conv1, ksize=2)

    with tf.name_scope('conv2'):
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = max_pool_2x2(conv2, ksize=2)

    with tf.name_scope('fc1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, DROPOUT)
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'], name="out")

    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, labels_size]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([labels_size]))
}


logits = model1(X, weights, biases, keep_prob)
predictions = tf.nn.softmax(logits)

with tf.name_scope('loss'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_op)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variable
init = tf.global_variables_initializer()

display_step = 100

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

# Start training
with tf.Session(config=config) as sess:

    sess.run(init)

    current_batch = 0

    for step in range(1, EPOCHS+1):

        # batch_x = train_x[current_batch: current_batch + BATCH_SIZE]
        # batch_y = train_y[current_batch: current_batch + BATCH_SIZE]

        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

        sess.run(train_op, feed_dict={X: batch_x, y: batch_y, keep_prob: 0.5})
        if step % display_step == 0 or step == 1:
            loss = sess.run(loss_op, feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
            print(f"Step {step}, training accuracy: {acc}")
        
    saver = tf.train.Saver()
    saver.save(sess, 'models/model1save/model1')

    test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    print(f"Testing accuracy: {test_acc}")
    print("optimization finished")
