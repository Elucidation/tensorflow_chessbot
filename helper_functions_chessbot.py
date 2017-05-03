# -*- coding: utf-8 -*-
# 
# Helper functions for the reddit chessbot
# Includes functions to parse FEN strings and get pithy quotes
import re
from helper_functions import lengthenFEN
from message_template import *

#########################################################
# ChessBot Message Generation Functions

def isPotentialChessboardTopic(sub):
  """if url is imgur link, or url ends in .png/.jpg/.gif"""
  if sub.url == None:
    return False
  return ('imgur' in sub.url
          or any([sub.url.lower().endswith(ending) for ending in ['.png', '.jpg', 'jpeg', '.gif']]))

def invert(fen):
  return ''.join(reversed(fen))

def generateMessage(fen, certainty, side, visualize_link):
  """Generate response message using FEN, certainty and side for flipping link order"""
  vals = {} # Holds template responses

  # Things that don't rely on black/white to play 
  # FEN image link is aligned with screenshot, not side to play
  if fen == '8/8/8/8/8/8/8/8':
    # Empty chessboard link, fen-to-image doesn't correctly identify those
    vals['unaligned_fen_img_link'] = 'http://i.stack.imgur.com/YxP53.gif'
  else:
    vals['unaligned_fen_img_link'] = 'http://www.fen-to-image.com/image/60/%s.png' % fen
  vals['certainty'] = certainty*100.0 # to percentage
  vals['pithy_message'] = getPithyMessage(certainty)
  
  if side == 'b':
    # Flip FEN if black to play, assumes image is flipped
    fen = invert(fen)
  
  inverted_fen = invert(fen)

  # Get castling status based on pieces being in initial positions or not
  castle_status = getCastlingStatus(fen)
  inverted_castle_status = getCastlingStatus(inverted_fen)

  # Fill out template and return
  vals['fen_w'] = "%s w %s -" % (fen, castle_status)
  vals['fen_b'] = "%s b %s -" % (fen, castle_status)
  vals['inverted_fen_w'] = "%s w %s -" % (inverted_fen, inverted_castle_status)
  vals['inverted_fen_b'] = "%s b %s -" % (inverted_fen, inverted_castle_status)

  vals['lichess_analysis_w'] = 'https://www.lichess.org/analysis/%s_w_%s' % (fen, castle_status)
  vals['lichess_analysis_b'] = 'https://www.lichess.org/analysis/%s_b_%s' % (fen, castle_status)
  vals['lichess_editor_w'] = 'https://www.lichess.org/editor/%s_w_%s' % (fen, castle_status)
  vals['lichess_editor_b'] = 'https://www.lichess.org/editor/%s_b_%s' % (fen, castle_status)

  vals['inverted_lichess_analysis_w'] = 'https://www.lichess.org/analysis/%s_w_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_analysis_b'] = 'https://www.lichess.org/analysis/%s_b_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_editor_w'] = 'https://www.lichess.org/editor/%s_w_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_editor_b'] = 'https://www.lichess.org/editor/%s_b_%s' % (inverted_fen, inverted_castle_status)

  vals['visualize_link'] = visualize_link
  
  return MESSAGE_TEMPLATE.format(**vals)



# Add a little message based on certainty of response
def getPithyMessage(certainty):
  pithy_messages = [
    '*[\[ ◕ _ ◕\]^*> ... \[⌐■ _ ■\]^*](http://i.imgur.com/yaVftzT.jpg)*',
    'A+ ✓',
    '✓',
    '[Close.](http://i.imgur.com/SwKKZlD.jpg)',
    '[WAI](http://gfycat.com/RightHalfIndianglassfish)',
    '[:(](http://i.imgur.com/BNwca4R.gifv)',
    '[I tried.](http://i.imgur.com/kmmp0lc.png)',
    '[Wow.](http://i.imgur.com/67fZDh9.webm)']
  pithy_messages_cutoffs = [0.999995, 0.99, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0]

  for cuttoff, pithy_message in zip(pithy_messages_cutoffs, pithy_messages):
    if certainty >= cuttoff:
      return pithy_message
  
  return ""

def getSideToPlay(title, fen):
  """Based on post title return 'w', 'b', or predict from FEN"""
  title = title.lower()
  # Return if 'black' in title unless 'white to' is, and vice versa, or predict if neither
  if 'black' in title:
    if 'white to' in title:
      return 'w'
    return 'b'
  elif 'white' in title:
    if 'black to' in title:
      return 'b'
    return 'w'
  else:
    # Predict side from fen (always returns 'w' or 'b', default 'w')
    return predictSideFromFEN(fen)

def predictSideFromFEN(fen):
  """Returns which side it thinks FEN is looking from.
     Checks number of white and black pieces on either side to determine
     i.e if more black pieces are on 1-4th ranks, then black to play"""

  # remove spaces values (numbers) from fen
  fen = re.sub('\d','',fen)
  
  #split fen to top half and bottom half (top half first)
  parts = fen.split('/')
  top = list(''.join(parts[:4]))
  bottom = list(''.join(parts[4:]))
  
  # If screenshot is aligned from POV of white to play, we'd expect
  # top to be mostly black pieces (lowercase)
  # and bottom to be mostly white pieces (uppercase), so lets count
  top_count_white = sum(list(map(lambda x: ord(x) <= ord('Z'), top)))
  bottom_count_white = sum(list(map(lambda x: ord(x) <= ord('Z'), bottom)))

  top_count_black = sum(list(map(lambda x: ord(x) >= ord('a'), top)))
  bottom_count_black = sum(list(map(lambda x: ord(x) >= ord('a'), bottom)))

  # If more white pieces on top side, or more black pieces on bottom side, black to play
  if (top_count_white > bottom_count_white or top_count_black < bottom_count_black):
    return 'b'

  # Otherwise white
  return 'w'

def getCastlingStatus(fen):
  """Check FEN to see if castling is allowed based on initial positions.
     Returns 'KQkq' variants or '-' if no castling."""
  
  fen = lengthenFEN(fen) # 71-char long fen
  # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR : initial position
  # 01234567                           01234567 +63

  status = ['','','',''] # KQkq
  # Check if black king can castle
  if fen[4] == 'k':
    # long (q)
    if fen[0] == 'r':
      status[3] = 'q'
    if fen[7] == 'r':
      status[2] = 'k'
  # Check if white king can castle
  if fen[63+4] == 'K':
    # long (Q)
    if fen[63+0] == 'R':
      status[1] = 'Q'
    if fen[63+7] == 'R':
      status[0] = 'K'
  
  status = ''.join(status)
  return status if status else '-'

def getFENtileLetter(fen,letter,number):
  """Given a fen string and a rank (number) and file (letter), return piece letter"""
  l2i = lambda l:  ord(l)-ord('A') # letter to index
  piece_letter = fen[(8-number)*8+(8-number) + l2i(letter)]
  return ' KQRBNPkqrbnp'.find(piece_letter)
