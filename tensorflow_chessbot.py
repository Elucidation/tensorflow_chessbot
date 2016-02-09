# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # TensorFlow Chessbot
# 
# The goal is to build a Reddit bot that listens on /r/chess for posts with an image in it (perhaps checking also for a statement "white/black to play" and an image link)
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
# A lot of tensorflow code here is heavily adopted from the [tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/pdes/index.html)

# <markdowncell>

# ---
# ## Start TF session

# <codecell>

import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)

sess = tf.InteractiveSession()

# <markdowncell>

# ## Load image
# 
# Let's first load a simple chessboard image taken off of reddit, we'll start simple, with the board filling up the entire space. Let's get the imports out of the way

# <codecell>

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy.signal
import os

def display_array(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = StringIO()
  PIL.Image.fromarray(a).save(f, fmt)
  #display(Image(data=f.getvalue()))

# <codecell>

# File
import glob
glob.glob("*.png")
img_files = set(glob.glob("*.png")).union(set(glob.glob("*.jpg"))).union(set(glob.glob("*.gif")))

# img_file = 'img1.png'
# img_file = 'img2.png'
# img_file = 'img3.gif'
# img_file = 'img4.jpg'
# img_file = 'img7.png'
# img_file = 'img9.png'
for img_file in img_files:
  img_save_dir = "squares_%s" % img_file[:-4]
  if os.path.exists(img_save_dir):
    print "Already processed %s, skipping..." % (img_save_dir)
    continue
  
  # Load image
  img = PIL.Image.open(img_file)


  # Resize
  img_size = 256
  img = img.resize([img_size,img_size], PIL.Image.ADAPTIVE)

  print "Loaded %s (%dpx x %dpx)" % \
      (img_file, img.size[0], img.size[1])
  # See original image
  display_array(np.asarray(img), rng=[0,255])

  # <codecell>

  # Convert to grayscale and array
  a = np.asarray(img.convert("L"), dtype=np.float32)

  # Display array
  display_array(a, rng=[0,255])

  # <markdowncell>

  # We need to find the chessboard squares within the image (assuming images will vary, boards will vary in color, etc. between different posts in reddit). A assumption we can make that simplifies things greatly is to assume the chessboards will be aligned with the image (orthorectified), so we only need to search for horizontal and vertical lines.
  # 
  # One way is to use horizontal and vertical gradients, and then a simplified hough transform on those gradient images to find the lines.

  # <codecell>

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

  def corners(x):
    """Find chess square corners in an array"""
    chess_corner = make_kernel([[-1., 0, 1],[0., 0., 0.], [1.,0, -1]])
    return simple_conv(x, chess_corner)

  # Following are meant for binary images
  def dilate(x, size=3):
    """Dilate"""
    kernel = make_kernel(np.ones([size,size], dtype=np.float32))
    return tf.clip_by_value(simple_conv(x, kernel),
                            np.float32(1),
                            np.float32(2))-np.float32(1)

  def erode(x, size=3):
    """Erode"""
    kernel = make_kernel(np.ones([size,size]))
    return tf.clip_by_value(simple_conv(x, kernel),
                            np.float32(size*size-1),
                            np.float32(size*size))-np.float32(size*size-1)

  def opening(x, size=3):
      return dilate(erode(x,size),size)

  def closing(x, size=3):
      return erode(dilate(x,size),size)

  def skeleton(x, size=3):
      """Skeletonize"""
      return tf.clip_by_value(erode(x) - opening(erode(x)),
                              0.,
                              1.)

  # <markdowncell>

  # Now that we've got our kernels ready for convolution, let's create our tf variables.

  # <codecell>

  # Get our grayscale image matrix
  A = tf.Variable(a)

  # Get X & Y gradients and subtract opposite gradient
  # Strongest response where gradient is unidirectional
  # clamp into range 0-1
  Dx = tf.clip_by_value(np.abs(gradientx(A)) - np.abs(gradienty(A)),
                        0., 1.)
  Dy = tf.clip_by_value(np.abs(gradienty(A)) - np.abs(gradientx(A)),
                        0., 1.)

  # Dxy = np.abs(gradientx(A) * gradienty(A))
  # Dc = np.abs(corners(A))


  # <codecell>

  # Initialize state to initial conditions
  tf.initialize_all_variables().run()

  # <markdowncell>

  # Let's look at the gradients, we apply opening to them also to clean up noise

  # <codecell>

  display_array(closing(Dx).eval())
  display_array(opening(Dy).eval())

  # <markdowncell>

  # Looks pretty good, now how to find lines? Well with a [Hough transform](https://en.wikipedia.org/wiki/Hough_transform) we resample into a parameter space of lines based on two variables $r$ and $\theta$ for example. In our case we already know we're doing vertical and horizontal lines so instead of a 2D space we just need two 1D spaces. In fact, we can simply do this by summing along the axes for each gradient.

  # <codecell>

  hough_Dx = tf.matmul(np.ones([1,a.shape[0]], dtype=np.float32), 
                       opening(Dx)) / a.shape[0]

  hough_Dy = tf.matmul(opening(Dy), 
                       np.ones([a.shape[1],1], dtype=np.float32)
                       ) / a.shape[1]

  # <markdowncell>

  # Let's plot the responses of the summed gradients

  # <codecell>
  peaks_x = hough_Dx.eval().flatten()
  peaks_y = hough_Dy.eval().flatten()

  # <codecell>

  # Blur
  gausswin = scipy.signal.gaussian(21,2)
  gausswin /= np.sum(gausswin)
  # plt.plot(gausswin)

  # Blur where there is a strong horizontal or vertical line (binarize)
  blur_x = np.convolve(peaks_x > .6, gausswin, mode='same')
  blur_y = np.convolve(peaks_y > .6, gausswin, mode='same')

  # Upsample before blur for sub-pixel peak
  # upsample_multiplier = np.float32(1)
  # blur_x_up = np.convolve(np.repeat(peaks_x > .6,upsample_multiplier), gausswin, mode='same')
  # blur_y_up = np.convolve(np.repeat(peaks_y > .6,upsample_multiplier), gausswin, mode='same')

  def skeletonize_1d(arr):
      _arr = arr.copy()
      """skeletonize an array (thin to single value, favor to the right)"""
      for i in range(_arr.size-1):
          # Will right-shift if they are the same
          if arr[i] <= _arr[i+1]:
              _arr[i] = 0
      
      for i in np.arange(_arr.size-1, 0,-1):
          if _arr[i-1] > _arr[i]:
              _arr[i] = 0
      return _arr


  skel_x = skeletonize_1d(blur_x)
  skel_y = skeletonize_1d(blur_y)

  # Find points on skeletonized arrays (where returns 1-length tuple)
  all_lines_x = np.where(skel_x)[0] # vertical lines
  all_lines_y = np.where(skel_y)[0] # horizontal lines


  # Remove points near edges
  edge_buffer = 25
  lines_x = all_lines_x[np.logical_and(all_lines_x > edge_buffer,
                                   all_lines_x < img_size-edge_buffer)]
  lines_y = all_lines_y[np.logical_and(all_lines_y > edge_buffer, all_lines_y < img_size-edge_buffer)]

  print "X",lines_x, np.diff(lines_x)
  print "Y",lines_y, np.diff(lines_y)

  # <codecell>

  # Plot blurred 1d hough arrays and skeletonized versions
  # <markdowncell>

  # Cool, we've got a set of lines now. We need to figure out which lines are associated with the chessboard, then split up the image into individual squares for feeding into the tensorflow CNN.

  # <codecell>

  print "X   (vertical)",lines_x, np.diff(lines_x)
  print "Y (horizontal)",lines_y, np.diff(lines_y)

  have_squares = False

  if lines_x.size == 7 and lines_y.size ==7:
      have_squares = True
      # Possibly check np.std(np.diff(lines_x)) for variance etc. as well/instead
      print "7 horizontal and vertical lines found, slicing up squares"
      
      # Find average square size, round to a whole pixel for determining edge pieces sizes
      stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
      stepy = np.int32(np.round(np.mean(np.diff(lines_y))))
      
      # Add outside sides for slicing image up
      np.hstack([lines_x[0]-stepx, lines_x, lines_x[-1]+stepx])
      setsx = np.hstack([max(0,lines_x[0]-stepx), lines_x, min(lines_x[-1]+stepx,img_size)])
      setsy = np.hstack([max(0,lines_y[0]-stepy), lines_y, min(lines_y[-1]+stepy, img_size)])
      
      print "X:",setsx
      print "Y:",setsy
      
      # Matrix to hold images of individual squares (in grayscale)
      squares = np.zeros([np.round(stepy), np.round(stepx), 64],dtype=np.uint8)
      
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
              # Apply padding as defined previously to fit
              squares[:,:,(7-j)*8+i] = np.pad(a[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
  else:
      print "Number of lines not equal to 7"

  # <codecell>

  letters = 'ABCDEFGH'

  if have_squares:
      print "Order is row-wise from top left of image going right and down, so a8,b8....a7,b7,c7...h1"
      print "Showing 5 random squares..."
      for i in np.random.choice(np.arange(64),5,replace=False):
          print "#%d: %s%d" % (i, letters[i%8], i/8+1)
          display_array(squares[:,:,i],rng=[0,255])
  else:
      print "Didn't have lines to slice image up."

  # <markdowncell>

  # Awesome! We have squares, let's save them as grayscale images in a subfolder with the same name as the image

  # <codecell>

  if not have_squares:
      print "No squares to save"
  else:
      if not os.path.exists(img_save_dir):
          os.makedirs(img_save_dir)
          print "Created dir %s" % img_save_dir
      
      for i in range(64):
          sqr_filename = "%s/%s_%s%d.png" % (img_save_dir, img_file[:-4], letters[i%8], i/8+1)
          print "#%d: saving %s..." % (i, sqr_filename)
        
          # Make resized 32x32 image from matrix and save
          PIL.Image.fromarray(squares[:,:,i]) \
              .resize([32,32], PIL.Image.ADAPTIVE) \
              .save(sqr_filename)

  # <codecell>


  # <codecell>


