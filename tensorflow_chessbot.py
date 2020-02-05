#!/usr/bin/env python3
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

from helper_functions import shortenFEN, unflipFEN
import helper_image_loading
import chessboard_finder

def load_graph(frozen_graph_filepath):
    # Load and parse the protobuf file to retrieve the unserialized graph_def.
    with tf.io.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph def and return.
    with tf.Graph().as_default() as graph:
        # Prefix every op/nodes in the graph.
        tf.import_graph_def(graph_def, name="tcb")
    return graph

class ChessboardPredictor(object):
  """ChessboardPredictor using saved model"""
  def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
    # Restore model using a frozen graph.
    print("\t Loading model '%s'" % frozen_graph_path)
    graph = load_graph(frozen_graph_path)
    self.sess = tf.compat.v1.Session(graph=graph)

    # Connect input/output pipes to model.
    self.x = graph.get_tensor_by_name('tcb/Input:0')
    self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
    self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
    self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')
    print("\t Model restored.")

  def getPrediction(self, tiles):
    """Run trained neural network on tiles generated from image"""
    if tiles is None or len(tiles) == 0:
      print("Couldn't parse chessboard")
      return None, 0.0
    
    # Reshape into Nx1024 rows of input data, format used by neural network
    validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)

    # Run neural network on data
    guess_prob, guessed = self.sess.run(
      [self.probabilities, self.prediction], 
      feed_dict={self.x: validation_set, self.keep_prob: 1.0})
    
    # Prediction bounds
    a = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed))))
    tile_certainties = a.reshape([8,8])[::-1,:]

    # Convert guess into FEN string
    # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
    labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
    pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed)) # exchange ' ' for '1' for FEN
    fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])
    return fen, tile_certainties

  ## Wrapper for chessbot
  def makePrediction(self, url):
    """Try and return a FEN prediction and certainty for URL, return Nones otherwise"""
    img, url = helper_image_loading.loadImageFromURL(url, max_size_bytes=2000000)
    result = [None, None, None]
    
    # Exit on failure to load image
    if img is None:
      print('Couldn\'t load URL: "%s"' % url)
      return result

    # Resize image if too large
    img = helper_image_loading.resizeAsNeeded(img)

    # Exit on failure if image was too large teo resize
    if img is None:
      print('Image too large to resize: "%s"' % url)
      return result

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
    args.url = None # Using filepath.
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
  predictor.close()
  if args.unflip:
      fen = unflipFEN(fen)
  short_fen = shortenFEN(fen)
  # Use the worst case certainty as our final uncertainty score
  certainty = tile_certainties.min()

  print('Per-tile certainty:')
  print(tile_certainties)
  print("Certainty range [%g - %g], Avg: %g" % (
    tile_certainties.min(), tile_certainties.max(), tile_certainties.mean()))

  active = args.active
  print("---\nPredicted FEN:\n%s %s - - 0 1" % (short_fen, active))
  print("Final Certainty: %.1f%%" % (certainty*100))

if __name__ == '__main__':
  np.set_printoptions(suppress=True, precision=3)
  import argparse
  parser = argparse.ArgumentParser(description='Predict a chessboard FEN from supplied local image link or URL')
  parser.add_argument('--url', default='http://imgur.com/u4zF5Hj.png', help='URL of image (ex. http://imgur.com/u4zF5Hj.png)')
  parser.add_argument('--filepath', help='filepath to image (ex. u4zF5Hj.png)')
  parser.add_argument('--unflip', default=False, action='store_true', help='revert the image of a flipped chessboard')
  parser.add_argument('--active', default='w')
  args = parser.parse_args()
  main(args)

  
