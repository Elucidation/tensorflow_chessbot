#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pass in image of online chessboard screenshot, returns corners of chessboard
# usage: chessboard_finder.py [-h] urls [urls ...]

# Find orthorectified chessboard corners in image

# positional arguments:
#   urls        Input image urls

# optional arguments:
#   -h, --help  show this help message and exit


# sudo apt-get install libatlas-base-dev for numpy error, see https://github.com/Kitt-AI/snowboy/issues/262
import numpy as np
# sudo apt-get install libopenjp2-7 libtiff5
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

  # shorten sequences to up to 9 values based on score
  # X sequences
  for i in range(len(seqs_x)):
    seq = seqs_x[i]
    seq_val = seqs_x_vals[i]

    # if the length of sequence is more than 7 + edges = 9
    # strip weakest edges 
    if len(seq) > 9:
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

  # Y sequences
  for i in range(len(seqs_y)):
    seq = seqs_y[i]
    seq_val = seqs_y_vals[i]

    while len(seq) > 9:
      if seq_val[0] > seq_val[-1]:
        seq = seq[:-1]
        seq_val = seq_val[:-1]
      else:
        seq = seq[1:]
        seq_val = seq_val[1:]

    seqs_y[i] = seq
    seqs_y_vals[i] = seq_val

  # Now that we only have length 7-9 sequences, score and choose the best one
  scores_x = np.array([np.mean(v) for v in seqs_x_vals])
  scores_y = np.array([np.mean(v) for v in seqs_y_vals])

  # Keep first sequence with the largest step size
  # scores_x = np.array([np.median(np.diff(s)) for s in seqs_x])
  # scores_y = np.array([np.median(np.diff(s)) for s in seqs_y])

  #TODO(elucidation): Choose heuristic score between step size and hough response

  best_seq_x = seqs_x[scores_x.argmax()]
  best_seq_y = seqs_y[scores_y.argmax()]
  # print(best_seq_x, best_seq_y)

  # Now if we have sequences greater than length 7, (up to 9),
  # that means we have up to 9 possible combinations of sets of 7 sequences
  # We try all of them and see which has the best checkerboard response
  sub_seqs_x = [best_seq_x[k:k+7] for k in range(len(best_seq_x) - 7 + 1)]
  sub_seqs_y = [best_seq_y[k:k+7] for k in range(len(best_seq_y) - 7 + 1)]

  dx = np.median(np.diff(best_seq_x))
  dy = np.median(np.diff(best_seq_y))
  corners = np.zeros(4, dtype=int)
  
  # Add 1 buffer to include the outer tiles, since sequences are only using
  # inner chessboard lines
  corners[0] = int(best_seq_y[0]-dy)
  corners[1] = int(best_seq_x[0]-dx)
  corners[2] = int(best_seq_y[-1]+dy)
  corners[3] = int(best_seq_x[-1]+dx)

  # Generate crop image with on full sequence, which may be wider than a normal
  # chessboard by an extra 2 tiles, we'll iterate over all combinations
  # (up to 9) and choose the one that correlates best with a chessboard
  gray_img_crop = PIL.Image.fromarray(img_arr_gray).crop(corners)

  # Build a kernel image of an idea chessboard to correlate against
  k = 8 # Arbitrarily chose 8x8 pixel tiles for correlation image
  quad = np.ones([k,k])
  kernel = np.vstack([np.hstack([quad,-quad]), np.hstack([-quad,quad])])
  kernel = np.tile(kernel,(4,4)) # Becomes an 8x8 alternating grid (chessboard)
  kernel = kernel/np.linalg.norm(kernel) # normalize
  # 8*8 = 64x64 pixel ideal chessboard

  k = 0
  n = max(len(sub_seqs_x), len(sub_seqs_y))
  final_corners = None
  best_score = None

  # Iterate over all possible combinations of sub sequences and keep the corners
  # with the best correlation response to the ideal 64x64px chessboard
  for i in range(len(sub_seqs_x)):
    for j in range(len(sub_seqs_y)):
      k = k + 1
      
      # [y, x, y, x]
      sub_corners = np.array([
        sub_seqs_y[j][0]-corners[0]-dy, sub_seqs_x[i][0]-corners[1]-dx,
        sub_seqs_y[j][-1]-corners[0]+dy, sub_seqs_x[i][-1]-corners[1]+dx],
        dtype=np.int)

      # Generate crop candidate, nearest pixel is fine for correlation check
      sub_img = gray_img_crop.crop(sub_corners).resize((64,64)) 

      # Perform correlation score, keep running best corners as our final output
      # Use absolute since it's possible board is rotated 90 deg
      score = np.abs(np.sum(kernel * sub_img))
      if best_score is None or score > best_score:
        best_score = score
        final_corners = sub_corners + [corners[0], corners[1], corners[0], corners[1]]

  return final_corners

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

def getChessBoardGray(img, corners):
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
  # Normalized
  chessboard_img_resized = np.asarray( \
        PIL.Image.fromarray(chessboard_img) \
        .resize([256,256], PIL.Image.BILINEAR), dtype=np.uint8) / 255.0
  return chessboard_img_resized

def getChessTilesGray(img, corners):
  chessboard_img_resized = getChessBoardGray(img, corners)
  return getTiles(chessboard_img_resized)


def getTiles(processed_gray_img):
  # Given 256x256 px normalized grayscale image of a chessboard (32x32px per tile)
  # NOTE (values must be in range 0-1)
  # Return a 32x32x64 tile array
  # 
  # stack deep 64 tiles
  # so, first slab is tile A1, then A2 etc.
  tiles = np.zeros([32,32,64], dtype=np.float32) # grayscale
  # Assume A1 is bottom left of image, need to reverse rank since images start
  # with origin in top left
  for rank in range(8): # rows (numbers)
    for file in range(8): # columns (letters)
      tiles[:,:,(rank*8+file)] = \
        processed_gray_img[(7-rank)*32:((7-rank)+1)*32,file*32:(file+1)*32]

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

# DEBUG
# from matplotlib import pyplot as plt
# def plotTiles(tiles):
#   """Plot color or grayscale tiles as 8x8 subplots"""
#   plt.figure(figsize=(6,6))
#   files = "ABCDEFGH"
#   for rank in range(8): # rows (numbers)
#     for file in range(8): # columns (letters)
#       plt.subplot(8,8,(7-rank)*8 + file + 1) # Plot rank reverse order to match image
      
#       if tiles.shape[2] == 64:
#         # Grayscale
#         tile = tiles[:,:,(rank*8+file)] # grayscale
#         plt.imshow(tile, interpolation='None', cmap='gray', vmin = 0, vmax = 1)
#       else:
#         #Color
#         tile = tiles[:,:,3*(rank*8+file):3*(rank*8+file+1)] # color
#         plt.imshow(tile, interpolation='None',)
      
#       plt.axis('off')
#       plt.title('%s %d' % (files[file], rank+1), fontsize=6)
#   plt.show()

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
    # tiles = getChessTilesGray(img_arr, corners)
    # plotTiles(tiles)

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

