#!/usr/bin/python
# -*- coding: utf-8 -*-
# Finds submissions with chessboard images in them,
# use a tensorflow convolutional neural network to predict pieces and return
# a lichess analysis link and FEN diagram of chessboard
import praw
import collections
import os
import time
from datetime import datetime
from praw.helpers import submission_stream
import requests
import socket
import re

import auth_config # for PRAW
import tensorflow_chessbot # For neural network model

#########################################################
# Setup

# Set up praw
chess_fen_bot = "ChessFenBot"
user_agent = chess_fen_bot + " finds chessboard image posts, uses convolutional neural network to responds with FEN diagram + analysis link. See https://github.com/Elucidation/tensorflow_chessbot"

# Login
r = praw.Reddit(user_agent=user_agent)

# Login old-style due to Reddit politics
r.login(auth_config.USERNAME, auth_config.PASSWORD, disable_warning=True)

# Get accessor to subreddit
subreddit = r.get_subreddit('chess')

# How many submissions to read from initially
submission_read_limit = 100

# How long to wait after replying to a post before continuing
reply_wait_time = 30 # minimum seconds to wait between replies, will also rate-limit safely

# Filename containing list of submission ids that 
# have already been processed, updated at end of program
processed_filename = "submissions_already_processed.txt"

# Submissions computer vision or prediction failed on
failures_filename = "submission_failures.txt"

# All responses id, fen + certainty
responses_filename = "submission_responses.txt"

# Response message template
message_template = """[◕ _ ◕]^*

I attempted to generate a [chessboard layout]({unaligned_fen_img_link}) from the posted image,
with a certainty of **{certainty:.4f}%**. *{pithy_message}*

* Link to [Lichess Analysis]({lichess_analysis})[^( Inverted)]({inverted_lichess_analysis}) - {to_play_full} to play
* FEN: `{fen}`

---

^(Yes I am a machine learning bot | )
[^(`How I work`)](https://github.com/Elucidation/tensorflow_chessbot 'Must go deeper')
^( | Reply with a corrected FEN or )[^(Editor)]({lichess_editor})
^(/)[^( Inverted)]({inverted_lichess_editor})^( to add to my next training dataset)

"""

#########################################################
# ChessBot Message Generation Functions

def isPotentialChessboardTopic(sub):
  """if url is imgur link, or url ends in .png/.jpg/.gif"""
  if sub.url == None:
    return False
  return ('imgur' in sub.url
          or any([sub.title.lower().endswith(ending) for ending in ['.png', '.jpg', '.gif']]))

def invert(fen):
  return ''.join(reversed(fen))

def generateMessage(fen, certainty, side):
  """Generate response message using FEN, certainty and side for flipping link order"""
  vals = {} # Holds template responses

  # Things that don't rely on black/white to play 
  # FEN image link is aligned with screenshot, not side to play
  vals['unaligned_fen_img_link'] = 'http://www.fen-to-image.com/image/30/%s.png' % fen
  vals['certainty'] = certainty*100 # to percentage
  vals['pithy_message'] = getPithyMessage(certainty)
  
  vals['to_play_full'] = 'White'
  if side == 'b':
    # Flip FEN if black to play, assumes image is flipped
    vals['to_play_full'] = 'Black'
    fen = invert(fen)
  
  # Fill out template and return
  vals['fen'] = fen
  vals['lichess_analysis'] = 'http://www.lichess.org/analysis/%s_%s' % (fen, side)
  vals['lichess_editor'] = 'http://www.lichess.org/editor/%s_%s' % (fen, side)
  vals['inverted_lichess_analysis'] = 'http://www.lichess.org/analysis/%s_%s' % (invert(fen), side)
  vals['inverted_lichess_editor'] = 'http://www.lichess.org/editor/%s_%s' % (invert(fen), side)
  return message_template.format(**vals)

# Add a little message based on certainty of response
pithy_messages = ['A+ ✓',
'✓',
'[Close.](http://i.imgur.com/SwKKZlD.jpg)',
'[WAI](http://gfycat.com/RightHalfIndianglassfish)',
'[I am ashamed.](http://i.imgur.com/BNwca4R.gifv)',
'[I tried.](http://i.imgur.com/kmmp0lc.png)',
'[Wow.](http://i.imgur.com/67fZDh9.webm)']
pithy_messages_cutoffs = [0.99, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0]
def getPithyMessage(certainty):
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



#########################################################
# PRAW Helper Functions

def waitWithComments(sleep_time, segment=60):
  """Sleep for sleep_time seconds, printing to stdout every segment of time"""
  print("\t%s - %s seconds to go..." % (datetime.now(), sleep_time))
  while sleep_time > segment:
    time.sleep(segment) # sleep in increments of 1 minute
    sleep_time -= segment
    print("\t%s - %s seconds to go..." % (datetime.now(), sleep_time))
  time.sleep(sleep_time)

def logInfoPerSubmission(submission, count, count_actual):
  if ((time.time() - logInfoPerSubmission.last) > 120):
    print("\n\t---\n\t%s - %d processed submissions, %d read\n" % (datetime.now(), count_actual, count))
    logInfoPerSubmission.last = time.time()
  try:
    print("#%d Submission(%s): %s" % (count, submission.id, submission))
  except UnicodeDecodeError as e:
    print("#%d Submission(%s): <ignoring unicode>" % (count, submission.id))


logInfoPerSubmission.last = time.time() # 'static' variable

def loadProcessed(processed_filename=processed_filename):
  if not os.path.isfile(processed_filename):
    print("%s - Starting new processed file" % datetime.now())
    return set()
  else:
    print("Loading existing processed file...")
    with open(processed_filename,'r') as f:
      return set([x.strip() for x in f.readlines()])

def saveProcessed(already_processed, processed_filename=processed_filename):
  with open(processed_filename,'w') as f:
    for submission_id in already_processed:
      f.write("%s\n" % submission_id)
  print("%s - Saved processed ids to file" % datetime.now())

def addSubmissionToFailures(submission, failures_filename=failures_filename):
  with open(failures_filename,'a') as f:
    f.write("%s : %s | %s\n" % (submission.id, submission.title, submission.url))
  print("%s - Saved failure to file" % datetime.now())  

def addSubmissionToResponses(submission, fen, certainty, side, responses_filename=responses_filename):
  # Reverse fen if it's black to play, assumes board is flipped  
  if side == 'b':
    fen = ''.join(reversed(fen))

  with open(responses_filename,'a') as f:
    f.write("%s : %s | %s | %s %s %g\n" % (submission.id, submission.title, submission.url, fen, side, certainty))
  print("%s - Saved response to file" % datetime.now())  

#########################################################
# Main Script
# Track commend ids that have already been processed successfully

# Load list of already processed comment ids
already_processed = loadProcessed()
print("%s - Starting with already processed: %s\n==========\n\n" % (datetime.now(), already_processed))

count = 0
count_actual = 0
running = True

# Start up Tensorflow CNN with trained model
predictor = tensorflow_chessbot.ChessboardPredictor()

while running:
  # get submission stream
  try:
    submissions = submission_stream(r, subreddit, limit=submission_read_limit)
    # for each submission
    for submission in submissions:
      count += 1
      # print out some debug info
      logInfoPerSubmission(submission, count, count_actual)

      # Skip if already processed
      if submission.id in already_processed:
        continue
      
      # check if submission title is a question
      if isPotentialChessboardTopic(submission):
        
        # Use CNN to make a prediction
        print "\n---\nImage URL: %s" % submission.url
        fen, certainty = predictor.makePrediction(submission.url)
        print "Predicted FEN: %s" % fen
        print "Certainty: %.4f%%" % (certainty*100)

        if fen is None:
          print("> %s - Couldn't generate FEN, skipping..." % datetime.now())
          # update & save list
          already_processed.add(submission.id)
          saveProcessed(already_processed)
          addSubmissionToFailures(submission)
          print "\n---\n"
          continue

        # Get side from title or fen
        side = getSideToPlay(submission.title, fen)
        # Generate response message
        msg = generateMessage(fen, certainty, side)
        print "fen: %s\nside: %s\n" % (fen, side)

        # respond, keep trying till success
        while True:
          try:
            print("> %s - Responding to %s: %s" % (datetime.now(), submission.id, submission))
            
            # Reply with comment
            submission.add_comment(msg)
            
            # update & save list
            already_processed.add(submission.id)
            saveProcessed(already_processed)
            addSubmissionToResponses(submission, fen, certainty, side)

            count_actual += 1
            print "\n---\n"
            # Wait after submitting to not overload
            waitWithComments(reply_wait_time)
            break
          except praw.errors.AlreadySubmitted as e:
            print("> %s - Already submitted skipping..." % datetime.now())
            break
          except praw.errors.RateLimitExceeded as e:
            print("> {} - Rate Limit Error for commenting on {}, sleeping for {} before retrying...".format(datetime.now(), submission.id, e.sleep_time))
            waitWithComments(e.sleep_time)

  
  # Handle errors
  except (socket.error, requests.exceptions.ReadTimeout, requests.packages.urllib3.exceptions.ReadTimeoutError, requests.exceptions.ConnectionError) as e:
    print("> %s - Connection error, resetting accessor, waiting 30 and trying again: %s" % (datetime.now(), e))
    saveProcessed(already_processed)
    time.sleep(30)
    continue
  except Exception as e:
    print("Unknown Error, continuing after 30:",e)
    time.sleep(30)
    continue
  except KeyboardInterrupt:
    print("Exiting...")
    running = False
  finally:
    saveProcessed(already_processed)
    print("%s - All Processed:\n%s" % (datetime.now(),already_processed))

print("%s - Program Ended. Total Processed Submissions (%d replied / %d read):\n%s" % (datetime.now(), count_actual, count, already_processed))
