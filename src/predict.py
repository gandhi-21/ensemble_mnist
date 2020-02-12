import tensorflow as tf
import pandas as pd
import numpy as np

# Create the network 
# Load the parameters

test = pd.read_csv("../input/test.csv")
test = test / 255.0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph("models/model1save/model1.meta")
    saver.restore(sess, tf.train.latest_checkpoint('models/model1save'))
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("fc1/out:0")
    x = graph.get_tensor_by_name("x:0")

    output1 = sess.run(output, feed_dict={x: test})

tf.reset_default_graph()

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph("models/model3save/model3.meta")
    saver.restore(sess, tf.train.latest_checkpoint('models/model3save'))
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("fc1/out:0")
    x = graph.get_tensor_by_name("x:0")

    output2 = sess.run(output, feed_dict={x: test})

sess.close()

with tf.Session(config=config) as sess:
    answer = (output1 + output2) / 2.0
    answer = tf.nn.softmax(answer)
    answers = answer.eval()
sess.close()

answers = np.argmax(answers, axis=1)

results = pd.Series(answers, name='Label')

submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)

submission.to_csv("sample.csv", index=False)