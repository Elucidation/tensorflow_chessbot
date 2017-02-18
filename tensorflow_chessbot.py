#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# TensorFlow Chessbot
# This contains ChessboardPredictor, the class responsible for loading and
# running a trained CNN on chessboard screenshots. Used by chessbot.py.
# A CLI interface is provided as well.
#
#   $ ./tensorflow_chessbot.py -h
#   usage: tensorflow_chessbot.py [-h] [--url URL] [--filepath FILEPATH]
# 
#    Predict a chessboard FEN from supplied local image link or URL
# 
#    optional arguments:
#      -h, --help           show this help message and exit
#      --url URL            URL of image (ex. http://imgur.com/u4zF5Hj.png)
#     --filepath FILEPATH  filepath to image (ex. u4zF5Hj.png)
# 
# This file is used by chessbot.py, a Reddit bot that listens on /r/chess for 
# posts with an image in it (perhaps checking also for a statement 
# "white/black to play" and an image link)
# 
# It then takes the image, uses some CV to find a chessboard on it, splits it up
# into a set of images of squares. These are the inputs to the tensorflow CNN
# which will return probability of which piece is on it (or empty)
# 
# Dataset will include chessboard squares from chess.com, lichess
# Different styles of each, all the pieces
# 
# Generate synthetic data via added noise:
#  * change in coloration
#  * highlighting
#  * occlusion from lines etc.
# 
# Take most probable set from TF response, use that to generate a FEN of the
# board, and bot comments on thread with FEN and link to lichess analysis.
# 
# A lot of tensorflow code here is heavily adopted from the 
# [tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/pdes/index.html)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Ignore Tensorflow INFO debug messages
import tensorflow as tf
import numpy as np

from helper_functions import shortenFEN
import helper_image_loading
import chessboard_finder

class ChessboardPredictor(object):
  """ChessboardPredictor using saved model"""
  def __init__(self, model_path='saved_models/model_10000.ckpt'):
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

    self.x = tf.placeholder(tf.float32, [None, 32*32])

    # First layer : 32 features
    W_conv1 = weight_variable([5, 5, 1, 32], name='W1')
    b_conv1 = bias_variable([32], name='B1')

    x_image = tf.reshape(self.x, [-1,32,32,1])

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
    self.keep_prob = tf.placeholder("float", name='KeepProb')
    h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob, name='Drop4')

    # Readout layer : softmax, 13 features
    W_fc2 = weight_variable([1024, 13], name='W5')
    b_fc2 = bias_variable([13], name='B5')

    self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='Ypredict')

    # Ground truth labels if exist 
    y_ = tf.placeholder(tf.float32, [None, 13], name='Ytruth')

    cross_entropy = -tf.reduce_sum(y_*tf.log(self.y_conv), name='CrossEntropy')

    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(y_,1), name='CorrectPrediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='Accuracy')

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start Interactive session for rest of notebook (else we'd want to close session)
    self.sess = tf.Session()
    tf.reset_default_graph() # clear out graph

    # Restore model from checkpoint
    print("\t Loading model '%s'" % model_path)
    saver.restore(self.sess, model_path)
    print("\t Model restored.")

  def getPrediction(self, tiles):
    """Run trained neural network on tiles generated from image"""
    if tiles is None or len(tiles) == 0:
      print("Couldn't parse chessboard")
      return None, 0.0
    
    # Reshape into Nx1024 rows of input data, format used by neural network
    validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)

    # Run neural network on data
    guess_prob, guessed = self.sess.run([self.y_conv, tf.argmax(self.y_conv,1)], feed_dict={self.x: validation_set, self.keep_prob: 1.0})
    
    # Prediction bounds
    a = np.array(map(lambda x: x[0][x[1]], zip(guess_prob, guessed)))
    tile_certainties = a.reshape([8,8])[::-1,:]

    # Convert guess into FEN string
    # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
    labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
    pieceNames = map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed) # exchange ' ' for '1' for FEN
    fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])
    return fen, tile_certainties

  ## Wrapper for chessbot
  def makePrediction(self, url):
    """Try and return a FEN prediction and certainty for URL, return Nones otherwise"""
    img, url = helper_image_loading.loadImageFromURL(url)
    result = [None, None, None]
    
    # Exit on failure to load image
    if img is None:
      print('Couldn\'t load URL: "%s"' % url)
      return result

    # Resize image if too large
    img = helper_image_loading.resizeAsNeeded(img)

    # Look for chessboard in image, get corners and split chessboard into tiles
    tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)

    # Exit on failure to find chessboard in image
    if tiles is None:
      print('Couldn\'t find chessboard in image')
      return result
    
    # Make prediction on input tiles
    fen, tile_certainties = self.getPrediction(tiles)
    
    # Use the worst case certainty as our final uncertainty score
    certainty = tile_certainties.min()

    # Get visualize link
    visualize_link = helper_image_loading.getVisualizeLink(corners, url)

    # Update result and return
    result = [fen, certainty, visualize_link]
    return result

  def close(self):
    print("Closing session.")
    self.sess.close()

###########################################################
# MAIN CLI

def main(args):
  # Load image from filepath or URL
  if args.filepath:
    # Load image from file
    img = helper_image_loading.loadImageFromPath(args.filepath)
  else:
    img, args.url = helper_image_loading.loadImageFromURL(args.url)

  # Exit on failure to load image
  if img is None:
    raise Exception('Couldn\'t load URL: "%s"' % args.url)
    
  # Resize image if too large
  # img = helper_image_loading.resizeAsNeeded(img)

  # Look for chessboard in image, get corners and split chessboard into tiles
  tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)

  # Exit on failure to find chessboard in image
  if tiles is None:
    raise Exception('Couldn\'t find chessboard in image')

  # Create Visualizer url link
  if args.url:
    viz_link = helper_image_loading.getVisualizeLink(corners, args.url)
    print('---\nVisualize tiles link:\n %s\n---' % viz_link)

  if args.url:
    print("\n--- Prediction on url %s ---" % args.url)
  else:
    print("\n--- Prediction on file %s ---" % args.filepath)
  
  # Initialize predictor, takes a while, but only needed once
  predictor = ChessboardPredictor()
  fen, tile_certainties = predictor.getPrediction(tiles)
  short_fen = shortenFEN(fen)
  # Use the worst case certainty as our final uncertainty score
  certainty = tile_certainties.min()

  print('Per-tile certainty:')
  print(tile_certainties)
  print("Certainty range [%g - %g], Avg: %g" % (
    tile_certainties.min(), tile_certainties.max(), tile_certainties.mean()))

  print("---\nPredicted FEN: %s" % short_fen)
  print("Final Certainty: %.1f%%" % (certainty*100))

if __name__ == '__main__':
  np.set_printoptions(suppress=True, precision=3)
  import argparse
  parser = argparse.ArgumentParser(description='Predict a chessboard FEN from supplied local image link or URL')
  parser.add_argument('--url', default='http://imgur.com/u4zF5Hj.png', help='URL of image (ex. http://imgur.com/u4zF5Hj.png)')
  parser.add_argument('--filepath', help='filepath to image (ex. u4zF5Hj.png)')
  args = parser.parse_args()
  main(args)

  