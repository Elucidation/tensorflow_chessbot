#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# TensorFlow Chessbot
# This is the main script for loading and running a trained CNN on chessboard 
# screenshots.
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
# It then takes the image, uses some CV to find a chessboard on it, splits up into
# a set of images of squares. These are the inputs to the tensorflow CNN
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

import tensorflow as tf
import numpy as np
import os
import glob
import argparse

# Imports for computer vision
import PIL.Image
import scipy.signal

import helper_functions

parser = argparse.ArgumentParser(description='Predict a chessboard FEN from supplied local image link or URL')
parser.add_argument('--url', help='URL of image (ex. http://imgur.com/u4zF5Hj.png)')
parser.add_argument('--filepath', help='filepath to image (ex. u4zF5Hj.png)')


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def gradientx(x):
  """Compute the x gradient of an array"""
  gradient_x = make_kernel([[-1.,0., 1.],
                            [-1.,0., 1.],
                            [-1.,0., 1.]])
  return simple_conv(x, gradient_x)

def gradienty(x):
  """Compute the x gradient of an array"""
  gradient_y = make_kernel([[-1., -1, -1],[0.,0,0], [1., 1, 1]])
  return simple_conv(x, gradient_y)

def checkMatch(lineset):
    """Checks whether there exists 7 lines of consistent increasing order in set of lines"""
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5

def pruneLines(lineset):
  """Prunes a set of lines to 7 in consistent increasing order (chessboard)"""
  linediff = np.diff(lineset)
  x = 0
  cnt = 0
  start_pos = 0
  for i, line in enumerate(linediff):
    # Within 5 px of the other (allowing for minor image errors)
    if np.abs(line - x) < 5:
      cnt += 1
      if cnt == 5:
        end_pos = i+2
        return lineset[start_pos:end_pos]
    else:
      cnt = 0
      x = line
      start_pos = i
  return []

def skeletonize_1d(arr):
  """return skeletonized 1d array (thin to single value, favor to the right)"""
  _arr = arr.copy() # create a copy of array to modify without destroying original
  # Go forwards
  for i in range(_arr.size-1):
      # Will right-shift if they are the same
      if arr[i] <= _arr[i+1]:
          _arr[i] = 0
  
  # Go reverse
  for i in np.arange(_arr.size-1, 0,-1):
      if _arr[i-1] > _arr[i]:
          _arr[i] = 0
  return _arr


gausswin = scipy.signal.gaussian(21,4)
gausswin /= np.sum(gausswin)

def getChessLines(hdx, hdy, hdx_thresh, hdy_thresh):
  """Returns pixel indices for the 7 internal chess lines in x and y axes"""

  # Blur where there is a strong horizontal or vertical line (binarize)
  blur_x = np.convolve((hdx > hdx_thresh)*1.0, gausswin, mode='same')
  blur_y = np.convolve((hdy > hdy_thresh)*1.0, gausswin, mode='same')

  skel_x = skeletonize_1d(blur_x)
  skel_y = skeletonize_1d(blur_y)

  # Find points on skeletonized arrays (where returns 1-length tuple)
  lines_x = np.where(skel_x)[0] # vertical lines
  lines_y = np.where(skel_y)[0] # horizontal lines
  
  # Prune inconsisten lines
  lines_x = pruneLines(lines_x)
  lines_y = pruneLines(lines_y)
  
  is_match = len(lines_x) == 7 and len(lines_y) == 7 and checkMatch(lines_x) and checkMatch(lines_y)
  
  return lines_x, lines_y, is_match

def getChessTiles(a, lines_x, lines_y):
  """Split up input grayscale array into 64 tiles stacked in a 3D matrix using the chess linesets"""
  # Find average square size, round to a whole pixel for determining edge pieces sizes

  stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
  stepy = np.int32(np.round(np.mean(np.diff(lines_y))))
  
  # Pad edges as needed to fill out chessboard (for images that are partially over-cropped)
  padr_x = 0
  padl_x = 0
  padr_y = 0
  padl_y = 0
  
  if lines_x[0] - stepx < 0:
    padl_x = np.abs(lines_x[0] - stepx)
  if lines_x[-1] + stepx > a.shape[1]-1:
    padr_x = np.abs(lines_x[-1] + stepx - a.shape[1])
  if lines_y[0] - stepy < 0:
    padl_y = np.abs(lines_y[0] - stepy)
  if lines_y[-1] + stepx > a.shape[0]-1:
    padr_y = np.abs(lines_y[-1] + stepy - a.shape[0])
  
  # New padded array
  a2 = np.pad(a, ((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
  
  setsx = np.hstack([lines_x[0]-stepx, lines_x, lines_x[-1]+stepx]) + padl_x
  setsy = np.hstack([lines_y[0]-stepy, lines_y, lines_y[-1]+stepy]) + padl_y
  
  a2 = a2[setsy[0]:setsy[-1], setsx[0]:setsx[-1]]
  setsx -= setsx[0]
  setsy -= setsy[0]

  # Tiles will contain 32x32x64 values corresponding to 64 chess tile images
  # A resize is needed to do this
  # tiles = np.zeros([np.round(stepy), np.round(stepx), 64],dtype=np.uint8)
  tiles = np.zeros([32, 32, 64],dtype=np.float32)
  
  # For each row
  for i in range(0,8):
    # For each column
    for j in range(0,8):
      # Vertical lines
      x1 = setsx[i]
      x2 = setsx[i+1]
      padr_x = 0
      padl_x = 0
      padr_y = 0
      padl_y = 0

      if (x2-x1) > stepx:
        if i == 7:
          x1 = x2 - stepx
        else:
          x2 = x1 + stepx
      elif (x2-x1) < stepx:
        if i == 7:
          # right side, pad right
          padr_x = stepx-(x2-x1)
        else:
          # left side, pad left
          padl_x = stepx-(x2-x1)
      # Horizontal lines
      y1 = setsy[j]
      y2 = setsy[j+1]

      if (y2-y1) > stepy:
        if j == 7:
          y1 = y2 - stepy
        else:
          y2 = y1 + stepy
      elif (y2-y1) < stepy:
        if j == 7:
          # right side, pad right
          padr_y = stepy-(y2-y1)
        else:
          # left side, pad left
          padl_y = stepy-(y2-y1)
      # slicing a, rows sliced with horizontal lines, cols by vertical lines so reversed
      # Also, change order so its A1,B1...H8 for a white-aligned board
      # Apply padding as defined previously to fit minor pixel offsets
      # tiles[:,:,(7-j)*8+i] = np.pad(a2[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
      full_size_tile = np.pad(a2[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
      tiles[:,:,(7-j)*8+i] = np.asarray( \
        PIL.Image.fromarray(full_size_tile) \
        .resize([32,32], PIL.Image.BILINEAR), dtype=np.float32) / 255.0
        #PIL.Image.ADAPTIVE causes image artifacts
  return tiles


def loadImage(img_file):
  """Load image from file, convert to grayscale float32 numpy array"""
  img = PIL.Image.open(img_file)
  img = resizeAsNeeded(img)

  # Convert to grayscale and return as an numpy array
  return np.asarray(img.convert("L"), dtype=np.float32)

def resizeAsNeeded(img):
  """Resize if image larger than 2k pixels on a side"""
  if img.size[0] > 2000 or img.size[1] > 2000:
    print("Image too big (%d x %d)" % (img.size[0], img.size[1]))
    new_size = 500.0 # px
    if img.size[0] > img.size[1]:
      # resize by width to new limit
      ratio = new_size / img.size[0]
    else:
      # resize by height
      ratio = new_size / img.size[1]
    print("Reducing by factor of %.2g" % (1./ratio))
    img = img.resize(img.size * ratio, PIL.Image.ADAPTIVE)
    print("New size: (%d x %d)" % (img.size[0], img.size[1]))
  return img

def getTiles(img_arr):
  """Find and slice 64 chess tiles from image in 3D Matrix"""
  # Get our grayscale image matrix
  A = tf.Variable(img_arr)


  # X & Y gradients
  Dx = gradientx(A)
  Dy = gradienty(A)


  Dx_pos = tf.clip_by_value(Dx, 0., 255., name="dx_positive")
  Dx_neg = tf.clip_by_value(Dx, -255., 0., name='dx_negative')
  Dy_pos = tf.clip_by_value(Dy, 0., 255., name="dy_positive")
  Dy_neg = tf.clip_by_value(Dy, -255., 0., name='dy_negative')

  # 1-D ampltitude of hough transform of gradients about X & Y axes
  # Chessboard lines have strong positive and negative gradients within an axis
  hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (img_arr.shape[0]*img_arr.shape[0])
  hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (img_arr.shape[1]*img_arr.shape[1])

  # Slightly closer to 3/5 threshold, since they're such strong responses
  hough_Dx_thresh = tf.reduce_max(hough_Dx) * 3/5
  hough_Dy_thresh = tf.reduce_max(hough_Dy) * 3/5

  # Transition from TensorFlow to normal values (todo, do TF right) 
  
  # Initialize A with image array input
  # tf.initialize_all_variables().run() # will reset CNN weights so be selective

  # Local tf session
  sess = tf.Session()
  sess.run(tf.initialize_variables([A], name='getTiles_init'))

  # Get chess lines (try a fiew sets)
  hdx, hdy, hdx_thresh, hdy_thresh = sess.run(
    [hough_Dx, hough_Dy, hough_Dx_thresh, hough_Dy_thresh])
  lines_x, lines_y, is_match = getChessLines(hdx, hdy, hdx_thresh, hdy_thresh)
  for percentage in np.array([0.9, 0.8, 0.7, 0.6]):
    if is_match:
      break
    else:
      print("Trying %d%% of threshold" % (100*percentage))
      lines_x, lines_y, is_match = getChessLines(hdx, hdy, 
        hdx_thresh * percentage, hdy_thresh * percentage)

  # Get the tileset
  if is_match:
    return getChessTiles(img_arr, lines_x, lines_y)
  else:
    print("\tNo Match, lines found (dx/dy):", lines_x, lines_y)
    return [] # No match, no tiles

def saveTiles(tiles, img_save_dir, img_file):
  letters = 'ABCDEFGH'
  if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
  
  for i in range(64):
    sqr_filename = "%s/%s_%s%d.png" % (img_save_dir, img_file, letters[i%8], i/8+1)
    
    # Make resized 32x32 image from matrix and save
    if tiles.shape != (32,32,64):
      PIL.Image.fromarray(tiles[:,:,i]) \
          .resize([32,32], PIL.Image.ADAPTIVE) \
          .save(sqr_filename)
    else:
      # Possibly saving floats 0-1 needs to change fromarray settings
      PIL.Image.fromarray((tiles[:,:,i]*255).astype(np.uint8)) \
          .save(sqr_filename)

def generateTileset(input_chessboard_folder, output_tile_folder):
  # Create output folder as needed
  if not os.path.exists(output_tile_folder):
    os.makedirs(output_tile_folder)

  # Get all image files of type png/jpg/gif
  img_files = set(glob.glob("%s/*.png" % input_chessboard_folder))\
    .union(set(glob.glob("%s/*.jpg" % input_chessboard_folder)))\
    .union(set(glob.glob("%s/*.gif" % input_chessboard_folder)))

  num_success = 0
  num_failed = 0
  num_skipped = 0

  for i, img_path in enumerate(img_files):
    print("#% 3d/%d : %s" % (i+1, len(img_files), img_path))
    # Strip to just filename
    img_file = img_path[len(input_chessboard_folder)+1:-4]

    # Create output save directory or skip this image if it exists
    img_save_dir = "%s/tiles_%s" % (output_tile_folder, img_file)
    
    if os.path.exists(img_save_dir):
      print("\tSkipping existing")
      num_skipped += 1
      continue
    
    # Load image
    print("---")
    print("Loading %s..." % img_path)
    img_arr = loadImage(img_path)

    # Get tiles
    print("\tGenerating tiles for %s..." % img_file)
    tiles = getTiles(img_arr)

    # Save tiles
    if len(tiles) > 0:
      print("\tSaving tiles %s" % img_file)
      saveTiles(tiles, img_save_dir, img_file)
      num_success += 1
    else:
      print("\tNo Match, skipping")
      num_failed += 1

  print("\t%d/%d generated, %d failures, %d skipped." % (num_success,
    len(img_files) - num_skipped, num_failed, num_skipped))

class ChessboardPredictor(object):
  """ChessboardPredictor using saved model"""
  def __init__(self, model_path='saved_models/model_10000.ckpt'):

    print("Setting up CNN TensorFlow graph...")
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

    # # Old single layer regression classifier
    # W = tf.Variable(tf.zeros([32*32, 13]))
    # b = tf.Variable(tf.zeros([13]))
    # y = tf.nn.softmax(tf.matmul(x, W) + b)

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

    # Restore model from checkpoint
    print("Loading model '%s'" % model_path)
    saver.restore(self.sess, model_path)
    print("Model restored.")

  def getPrediction(self,img):
    """Run trained neural network on tiles generated from image"""
    
    # Convert to grayscale numpy array
    img_arr = np.asarray(img.convert("L"), dtype=np.float32)
    
    # Use computer vision to get the tiles
    tiles = getTiles(img_arr)
    if tiles is None or len(tiles) == 0:
      print("Couldn't parse chessboard")
      return None, 0.0
    
    # Reshape into Nx1024 rows of input data, format used by neural network
    validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)

    # Run neural network on data
    guess_prob, guessed = self.sess.run([self.y_conv, tf.argmax(self.y_conv,1)], feed_dict={self.x: validation_set, self.keep_prob: 1.0})
    
    # Prediction bounds
    a = np.array(map(lambda x: x[0][x[1]], zip(guess_prob, guessed)))
    print("Certainty range [%g - %g], Avg: %g, Overall: %g" % (a.min(), a.max(), a.mean(), a.prod()))
    
    # Convert guess into FEN string
    # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
    pieceNames = map(lambda k: '1' if k == 0 else helper_functions.labelIndex2Name(k), guessed) # exchange ' ' for '1' for FEN
    fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])
    return fen, a.prod()

  #########################################################
  # CNN Prediction Wrapper

  def makePrediction(self,image_url):
    """Return FEN prediction, and certainty for a URL"""
    # Try to load image url
    img = helper_functions.loadImageURL(image_url)

    if img == None:
      print("Couldn't load image url: %s" % image_url)
      return None, 0.0
    
    # Make prediction
    fen, certainty = self.getPrediction(img)
    if fen:
      return fen, certainty
    else:
      return None, 0.0

  def makePredictionFromFile(self,image_path):
    """Return FEN prediction, and certainty for a image file"""
    # Try to load image url
    img = helper_functions.loadImageFromPath(image_path)

    if img == None:
      print("Couldn't load image path: %s" % image_path)
      return None, 0.0
    
    # Make prediction
    fen, certainty = self.getPrediction(img)
    if fen:
      return fen, certainty
    else:
      return None, 0.0

###########################################################
# MAIN

if __name__ == '__main__':
  args = parser.parse_args()
  
  # Initialize predictor, takes a while, but only needed once
  predictor = ChessboardPredictor()

  if args.filepath:
    fen, certainty = predictor.makePredictionFromFile(args.filepath)
    print("Predicted FEN: %s" % fen)
    print("Certainty: %.1f%%" % (certainty*100))
  else:
    if args.url:
      url = args.url
    else:
      url = 'http://imgur.com/u4zF5Hj.png'

    fen, certainty = predictor.makePrediction(url)
    print("Predicted FEN: %s" % fen)
    print("Certainty: %.1f%%" % (certainty*100))

  print("Done")