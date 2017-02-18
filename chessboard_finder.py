#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pass in image of online chessboard screenshot, returns corners of chessboard
# usage: chessboard_finder.py [-h] urls [urls ...]

# Find orthorectified chessboard corners in image

# positional arguments:
#   urls        Input image urls

# optional arguments:
#   -h, --help  show this help message and exit


import numpy as np
import PIL.Image
import argparse
from time import time
from helper_image_loading import *


def nonmax_suppress_1d(arr, winsize=5):
  """Return 1d array with only peaks, use neighborhood window of winsize px"""
  _arr = arr.copy()

  for i in range(_arr.size):
    if i == 0:
      left_neighborhood = 0
    else:
      left_neighborhood = arr[max(0,i-winsize):i]
    if i >= _arr.size-2:
      right_neighborhood = 0
    else:
      right_neighborhood = arr[i+1:min(arr.size-1,i+winsize)]

    if arr[i] < np.max(left_neighborhood) or arr[i] <= np.max(right_neighborhood):
      _arr[i] = 0
  return _arr

def findChessboardCorners(img_arr_gray, noise_threshold = 8000):
  # Load image grayscale as an numpy array
  # Return None on failure to find a chessboard
  #
  # noise_threshold: Ratio of standard deviation of hough values along an axis
  # versus the number of pixels, manually measured  bad trigger images
  # at < 5,000 and good  chessboards values at > 10,000

  # Get gradients, split into positive and inverted negative components 
  gx, gy = np.gradient(img_arr_gray)
  gx_pos = gx.copy()
  gx_pos[gx_pos<0] = 0
  gx_neg = -gx.copy()
  gx_neg[gx_neg<0] = 0

  gy_pos = gy.copy()
  gy_pos[gy_pos<0] = 0
  gy_neg = -gy.copy()
  gy_neg[gy_neg<0] = 0

  # 1-D ampltitude of hough transform of gradients about X & Y axes
  num_px = img_arr_gray.shape[0] * img_arr_gray.shape[1]
  hough_gx = gx_pos.sum(axis=1) * gx_neg.sum(axis=1)
  hough_gy = gy_pos.sum(axis=0) * gy_neg.sum(axis=0)

  # Check that gradient peak signal is strong enough by
  # comparing normalized standard deviation to threshold
  if min(hough_gx.std() / hough_gx.size,
         hough_gy.std() / hough_gy.size) < noise_threshold:
    return None
  
  # Normalize and skeletonize to just local peaks
  hough_gx = nonmax_suppress_1d(hough_gx) / hough_gx.max()
  hough_gy = nonmax_suppress_1d(hough_gy) / hough_gy.max()

  # Arbitrary threshold of 20% of max
  hough_gx[hough_gx<0.2] = 0
  hough_gy[hough_gy<0.2] = 0

  # Now we have a set of potential vertical and horizontal lines that
  # may contain some noisy readings, try different subsets of them with
  # consistent spacing until we get a set of 7, choose strongest set of 7
  pot_lines_x = np.where(hough_gx)[0]
  pot_lines_y = np.where(hough_gy)[0]
  pot_lines_x_vals = hough_gx[pot_lines_x]
  pot_lines_y_vals = hough_gy[pot_lines_y]

  # Get all possible length 7+ sequences
  seqs_x = getAllSequences(pot_lines_x)
  seqs_y = getAllSequences(pot_lines_y)
  
  if len(seqs_x) == 0 or len(seqs_y) == 0:
    return None
  
  # Score sequences by the strength of their hough peaks
  seqs_x_vals = [pot_lines_x_vals[[v in seq for v in pot_lines_x]] for seq in seqs_x]
  seqs_y_vals = [pot_lines_y_vals[[v in seq for v in pot_lines_y]] for seq in seqs_y]

  for i in range(len(seqs_x)):
    seq = seqs_x[i]
    seq_val = seqs_x_vals[i]

    # while not inner 7 chess lines, strip weakest edges
    while len(seq) > 7:
      if seq_val[0] > seq_val[-1]:
        seq = seq[:-1]
        seq_val = seq_val[:-1]
      else:
        seq = seq[1:]
        seq_val = seq_val[1:]

    seqs_x[i] = seq
    seqs_x_vals[i] = seq_val

  for i in range(len(seqs_y)):
    seq = seqs_y[i]
    seq_val = seqs_y_vals[i]

    # while not inner 7 chess lines, strip weakest edges
    while len(seq) > 7:
      if seq_val[0] > seq_val[-1]:
        seq = seq[:-1]
        seq_val = seq_val[:-1]
      else:
        seq = seq[1:]
        seq_val = seq_val[1:]

    seqs_y[i] = seq
    seqs_y_vals[i] = seq_val

  # Now that we only have length 7 sequences, score and choose the best one
  scores_x = np.array([np.sum(v) for v in seqs_x_vals])
  scores_y = np.array([np.sum(v) for v in seqs_y_vals])

  best_seq_x = seqs_x[scores_x.argmax()]
  best_seq_y = seqs_y[scores_y.argmax()]

  # plt.figure(1)
  # plt.subplot(211)
  # plt.plot(hough_gx,'.-')
  # plt.plot(best_seq_x, np.ones(7)*0.5,'o')
  # plt.subplot(212)
  # plt.plot(hough_gy,'.-')
  # plt.plot(best_seq_y, np.ones(7)*0.5,'o')

  corners = np.zeros(4, dtype=int)
  
  # Invert row col to match x/y
  corners[0] = int(best_seq_y[0]-(best_seq_y[-1]-best_seq_y[0])/6)
  corners[1] = int(best_seq_x[0]-(best_seq_x[-1]-best_seq_x[0])/6)
  
  corners[2] = int(best_seq_y[-1]+(best_seq_y[-1]-best_seq_y[0])/6)
  corners[3] = int(best_seq_x[-1]+(best_seq_x[-1]-best_seq_x[0])/6)

  return corners

def getAllSequences(seq, min_seq_len=7, err_px=5):
  """Given sequence of increasing numbers, get all sequences with common
  spacing (within err_px) that contain at least min_seq_len values"""

  # Sanity check that there are enough values to satisfy
  if len(seq) < min_seq_len:
    return []

  # For every value, take the next value and see how many times we can step
  # that falls on another value within err_px points
  seqs = []
  for i in range(len(seq)-1):
    for j in range(i+1, len(seq)):
      # Check that seq[i], seq[j] not already in previous sequences
      duplicate = False
      for prev_seq in seqs:
        for k in range(len(prev_seq)-1):
          if seq[i] == prev_seq[k] and seq[j] == prev_seq[k+1]:
            duplicate = True
      if duplicate:
        continue
      d = seq[j] - seq[i]
      
      # Ignore two points that are within error bounds of each other
      if d < err_px:
        continue

      s = [seq[i], seq[j]]
      n = s[-1] + d
      while np.abs((seq-n)).min() < err_px:
        n = seq[np.abs((seq-n)).argmin()]
        s.append(n)
        n = s[-1] + d

      if len(s) >= min_seq_len:
        s = np.array(s)
        seqs.append(s)
  return seqs

def getChessTilesColor(img, corners):
  # img is a color RGB image
  # corners = (x0, y0, x1, y1) for top-left corner to bot-right corner of board
  height, width, depth = img.shape
  if depth !=3:
    print("Need RGB color image input")
    return None

  # corners could be outside image bounds, pad image as needed
  padl_x = max(0, -corners[0])
  padl_y = max(0, -corners[1])
  padr_x = max(0, corners[2] - width)
  padr_y = max(0, corners[3] - height)

  img_padded = np.pad(img, ((padl_y,padr_y),(padl_x,padr_x), (0,0)), mode='edge')

  chessboard_img = img_padded[
    (padl_y + corners[1]):(padl_y + corners[3]), 
    (padl_x + corners[0]):(padl_x + corners[2]), :]

  # 256x256 px RGB image, 32x32px individual RGB tiles, normalized 0-1 floats
  chessboard_img_resized = np.asarray( \
        PIL.Image.fromarray(chessboard_img) \
        .resize([256,256], PIL.Image.BILINEAR), dtype=np.float32) / 255.0

  # stack deep 64 tiles with 3 channesl RGB each
  # so, first 3 slabs are RGB for tile A1, then next 3 slabs for tile A2 etc.
  tiles = np.zeros([32,32,3*64], dtype=np.float32) # color
  # Assume A1 is bottom left of image, need to reverse rank since images start
  # with origin in top left
  for rank in range(8): # rows (numbers)
    for file in range(8): # columns (letters)
      # color
      tiles[:,:,3*(rank*8+file):3*(rank*8+file+1)] = \
        chessboard_img_resized[(7-rank)*32:((7-rank)+1)*32,file*32:(file+1)*32]

  return tiles

def getChessTilesGray(img, corners):
  # img is a grayscale image
  # corners = (x0, y0, x1, y1) for top-left corner to bot-right corner of board
  height, width = img.shape

  # corners could be outside image bounds, pad image as needed
  padl_x = max(0, -corners[0])
  padl_y = max(0, -corners[1])
  padr_x = max(0, corners[2] - width)
  padr_y = max(0, corners[3] - height)

  img_padded = np.pad(img, ((padl_y,padr_y),(padl_x,padr_x)), mode='edge')

  chessboard_img = img_padded[
    (padl_y + corners[1]):(padl_y + corners[3]), 
    (padl_x + corners[0]):(padl_x + corners[2])]

  # 256x256 px image, 32x32px individual tiles
  chessboard_img_resized = np.asarray( \
        PIL.Image.fromarray(chessboard_img) \
        .resize([256,256], PIL.Image.BILINEAR), dtype=np.uint8) / 255.0
  
  # stack deep 64 tiles
  # so, first slab is tile A1, then A2 etc.
  tiles = np.zeros([32,32,64], dtype=np.float32) # grayscale
  # Assume A1 is bottom left of image, need to reverse rank since images start
  # with origin in top left
  for rank in range(8): # rows (numbers)
    for file in range(8): # columns (letters)
      tiles[:,:,(rank*8+file)] = \
        chessboard_img_resized[(7-rank)*32:((7-rank)+1)*32,file*32:(file+1)*32]

  return tiles

def findGrayscaleTilesInImage(img):
  """ Find chessboard and convert into input tiles for CNN """
  if img is None:
    return None, None

  # Convert to grayscale numpy array 
  img_arr = np.asarray(img.convert("L"), dtype=np.float32)
  
  # Use computer vision to find orthorectified chessboard corners in image
  corners = findChessboardCorners(img_arr)
  if corners is None:
    return None, None

  # Pull grayscale tiles out given image and chessboard corners
  tiles = getChessTilesGray(img_arr, corners)

  # Return both the tiles as well as chessboard corner locations in the image
  return tiles, corners

def plotTiles(tiles):
  """Plot color or grayscale tiles as 8x8 subplots"""
  from matplotlib import pyplot as plt
  plt.figure(figsize=(6,6))
  files = "ABCDEFGH"
  for rank in range(8): # rows (numbers)
    for file in range(8): # columns (letters)
      plt.subplot(8,8,(7-rank)*8 + file + 1) # Plot rank reverse order to match image
      
      if tiles.shape[2] == 64:
        # Grayscale
        tile = tiles[:,:,(rank*8+file)] # grayscale
        plt.imshow(tile, interpolation='None', cmap='gray', vmin = 0, vmax = 1)
      else:
        #Color
        tile = tiles[:,:,3*(rank*8+file):3*(rank*8+file+1)] # color
        plt.imshow(tile, interpolation='None',)
      
      plt.axis('off')
      plt.title('%s %d' % (files[file], rank+1), fontsize=6)
  plt.show()

def main(url):
  print("Loading url %s..." % url)
  color_img, url = loadImageFromURL(url)
  
  # Fail if can't load image
  if color_img is None:
    print('Couldn\'t load url: %s' % url)
    return

  if color_img.mode != 'RGB':
    color_img = color_img.convert('RGB')
  print("Processing...")
  a = time()
  img_arr = np.asarray(color_img.convert("L"), dtype=np.float32)
  corners = findChessboardCorners(img_arr)
  print("Took %.4fs" % (time()-a))
  # corners = [x0, y0, x1, y1] where (x0,y0) 
  # is top left and (x1,y1) is bot right

  if corners is not None:
    print("\tFound corners for %s: %s" % (url, corners))
    link = getVisualizeLink(corners, url)
    print(link)

    # tiles = getChessTilesColor(np.array(color_img), corners)
    tiles = getChessTilesGray(img_arr, corners)
    plotTiles(tiles)

    # plt.imshow(color_img, interpolation='none')
    # plt.plot(corners[[0,0,2,2,0]]-0.5, corners[[1,3,3,1,1]]-0.5, color='red', linewidth=1)
    # plt.show()
  else:
    print('\tNo corners found in image')

if __name__ == '__main__':
  np.set_printoptions(suppress=True, precision=2)
  parser = argparse.ArgumentParser(description='Find orthorectified chessboard corners in image')
  parser.add_argument('urls', default=['https://i.redd.it/1uw3h772r0fy.png'],
    metavar='urls', type=str,  nargs='*', help='Input image urls')
  # main('http://www.chessanytime.com/img/jeudirect/simplechess.png')
  # main('https://i.imgur.com/JpzfV3y.jpg')
  # main('https://i.imgur.com/jsCKzU9.jpg')
  # main('https://i.imgur.com/49htmMA.png')
  # main('https://i.imgur.com/HHdHGBX.png')
  # main('http://imgur.com/By2xJkO')
  # main('http://imgur.com/p8DJMly')
  # main('https://i.imgur.com/Ns0iBrw.jpg')
  # main('https://i.imgur.com/KLcCiuk.jpg')
  args = parser.parse_args()
  for url in args.urls:
    main(url)

