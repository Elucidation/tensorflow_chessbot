# -*- coding: utf-8 -*-
# 
# Helper functions for the reddit chessbot
# Includes functions to parse/manipulate FEN strings and get pithy quotes
import re

def shortenFEN(fen):
  """Reduce FEN to shortest form (ex. '111p11Q' becomes '3p2Q')"""
  return fen.replace('11111111','8').replace('1111111','7') \
            .replace('111111','6').replace('11111','5') \
            .replace('1111','4').replace('111','3').replace('11','2')

def lengthenFEN(fen):
  """Lengthen FEN to 71-character form (ex. '3p2Q' becomes '111p11Q')"""
  return fen.replace('8','11111111').replace('7','1111111') \
            .replace('6','111111').replace('5','11111') \
            .replace('4','1111').replace('3','111').replace('2','11')

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