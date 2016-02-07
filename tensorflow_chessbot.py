# Reddit bot that listens on /r/chess for posts with an image in it
# (perhaps as well as a statement "white/black to play"
# and an image link
#
# It takes the image, uses some CV to find a chessboard on it, splits up into
# a set of images of squares. These are the inputs to the tensorflow CNN
# which will return probability of which piece is on it (or empty)
# 
# dataset will include chessboard squares from chess.com, lichess
# Different styles of each, all the pieces
# Generate synthetic data via added noise:
#  * change in coloration
#  * highlighting
#  * occlusion from lines etc.
#
# Take most probable set from TF response, use that to generate a FEN of the
# board, and bot comments on thread with FEN and link to lichess analysis

import tensorflow as tf

hello = tf.constant('Hello, TF!')
sess = tf.Session()
print sess.run(hello)
a = tf.constant([[1., 2.]])
b = tf.constant([[2.], [5.]])
c = tf.matmul(a,b)
print sess.run(a*b), sess.run(c)
