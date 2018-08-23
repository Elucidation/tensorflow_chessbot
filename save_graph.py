#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate graph.pb and graph.pbtxt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Ignore Tensorflow INFO debug messages
import tensorflow as tf
import numpy as np

# Output graph to the same directory as the checkpoint.
output_graph = "saved_models/graph.pb"
output_graphtxt = ('saved_models', 'graph.pbtxt')

# Set up a fresh session and create the model and load it from the saved checkpoint.
tf.reset_default_graph() # clear out graph.
sess = tf.Session()

model_path='saved_models/model_10000.ckpt'

def weight_variable(shape, name=""):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name=""):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name=""):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

x = tf.placeholder(tf.float32, [None, 32*32], 'Input')

# First layer : 32 features
W_conv1 = weight_variable([5, 5, 1, 32], name='W1')
b_conv1 = bias_variable([32], name='B1')

x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='Conv1')
h_pool1 = max_pool_2x2(h_conv1, name='Pool1')

# Second convolutional layer : 64 features
W_conv2 = weight_variable([5, 5, 32, 64], name='W2')
b_conv2 = bias_variable([64], name='B2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='Conv2')
h_pool2 = max_pool_2x2(h_conv2, name='Pool2')

# Densely connected layer : 1024 neurons, image size now 8x8
W_fc1 = weight_variable([8 * 8 * 64, 1024], name='W3')
b_fc1 = bias_variable([1024], name='B3')

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64], name='Pool3')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, 'MatMult3')

# Dropout
keep_prob = tf.placeholder("float", name='KeepProb')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='Drop4')

# Readout layer : softmax, 13 features
W_fc2 = weight_variable([1024, 13], name='W5')
b_fc2 = bias_variable([13], name='B5')

# Probabilities
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='probabilities')

# Final prediction
prediction = tf.argmax(y_conv,1, name='prediction')

# Ground truth labels if exist 
y_ = tf.placeholder(tf.float32, [None, 13], name='Ytruth')
actual_answer = tf.argmax(y_,1, name='actual')

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name='CrossEntropy')

# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(prediction, actual_answer, name='CorrectPrediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='Accuracy')

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Restore model from checkpoint
print("\t Loading model '%s'" % model_path)
saver.restore(sess, model_path)
print("\t Model restored.")

# Write graph in text format
tf.train.write_graph(sess.graph_def,output_graphtxt[0], output_graphtxt[1])

# To freeze graph then use:
# python3 -m tensorflow.python.tools.freeze_graph --input_graph graph.pbtxt --input_checkpoint=model_10000.ckpt  --input_binary=false --output_graph=actual_frozen.pb --output_node_names=prediction,probabilities

# We also save the binary-encoded graph that may or may not be frozen (TBD) below.
# We use a built-in TF helper to export variables to constants
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, # The session is used to retrieve the weights
    tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
    ["prediction", "probabilities"] # The output node names are used to select the useful nodes
) 

# Finally we serialize and dump the output graph to the filesystem
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print("%d ops in the final graph." % len(output_graph_def.node))