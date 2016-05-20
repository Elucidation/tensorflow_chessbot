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
from helper_functions_chessbot import *
import auth_config # for PRAW
import tensorflow_chessbot # For neural network model

#########################################################
# Setup

# Set up praw
chess_fen_bot = "ChessFenBot"

# Login
r = praw.Reddit(auth_config.USER_AGENT) 

# Login old-style due to Reddit politics
r.login(auth_config.USERNAME, auth_config.PASSWORD, disable_warning=True)

# Get accessor to subreddit
subreddit = r.get_subreddit('chess+chessbeginners+AnarchyChess+betterchess')

# How many submissions to read from initially
submission_read_limit = 100

# How long to wait after replying to a post before continuing
reply_wait_time = 10 # minimum seconds to wait between replies, will also rate-limit safely

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
with a certainty of **{certainty:.3f}%**. *{pithy_message}*

-

◇ White to play : [Analysis]({lichess_analysis_w}) | [Editor]({lichess_editor_w}) 
`{fen_w}`

-

◆ Black to play : [Analysis]({lichess_analysis_b}) | [Editor]({lichess_editor_b})
`{fen_b}`

-

> ▾ Links for when pieces are inverted on the board:
> 
> White to play : [Analysis]({inverted_lichess_analysis_w}) | [Editor]({inverted_lichess_editor_w})
> `{inverted_fen_w}`
>
> Black to play : [Analysis]({inverted_lichess_analysis_b}) | [Editor]({inverted_lichess_editor_b})
> `{inverted_fen_b}`

-

---

^(Yes I am a machine learning bot | )
[^(`How I work`)](http://github.com/Elucidation/tensorflow_chessbot 'Must go deeper')
^( | Reply with a corrected FEN to add to my next training dataset)

"""

#########################################################
# ChessBot Message Generation Functions

def isPotentialChessboardTopic(sub):
  """if url is imgur link, or url ends in .png/.jpg/.gif"""
  if sub.url == None:
    return False
  return ('imgur' in sub.url
          or any([sub.url.lower().endswith(ending) for ending in ['.png', '.jpg', '.gif']]))

def invert(fen):
  return ''.join(reversed(fen))

def generateMessage(fen, certainty, side):
  """Generate response message using FEN, certainty and side for flipping link order"""
  vals = {} # Holds template responses

  # Things that don't rely on black/white to play 
  # FEN image link is aligned with screenshot, not side to play
  vals['unaligned_fen_img_link'] = 'http://www.fen-to-image.com/image/30/%s.png' % fen
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

  vals['lichess_analysis_w'] = 'http://www.lichess.org/analysis/%s_w_%s' % (fen, castle_status)
  vals['lichess_analysis_b'] = 'http://www.lichess.org/analysis/%s_b_%s' % (fen, castle_status)
  vals['lichess_editor_w'] = 'http://www.lichess.org/editor/%s_w_%s' % (fen, castle_status)
  vals['lichess_editor_b'] = 'http://www.lichess.org/editor/%s_b_%s' % (fen, castle_status)

  vals['inverted_lichess_analysis_w'] = 'http://www.lichess.org/analysis/%s_w_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_analysis_b'] = 'http://www.lichess.org/analysis/%s_b_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_editor_w'] = 'http://www.lichess.org/editor/%s_w_%s' % (inverted_fen, inverted_castle_status)
  vals['inverted_lichess_editor_b'] = 'http://www.lichess.org/editor/%s_b_%s' % (inverted_fen, inverted_castle_status)
  
  return message_template.format(**vals)




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

def logInfoPerSubmission(submission, count, count_actual, is_processed=False):
  if ((time.time() - logInfoPerSubmission.last) > 120):
    print("\n\t---\n\t%s - %d processed submissions, %d read\n" % (datetime.now(), count_actual, count))
    logInfoPerSubmission.last = time.time()
  is_proc = ''
  if is_processed:
    is_proc = ' P'
  try:
    print("#%d Submission(%s%s): %s" % (count, submission.id, is_proc, submission))
  except UnicodeDecodeError as e:
    print("#%d Submission(%s%s): <ignoring unicode>" % (count, submission.id, is_proc))


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
print("%s - Starting with %d already processed\n==========\n\n" % (datetime.now(), len(already_processed)))

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
      is_processed = submission.id in already_processed
      logInfoPerSubmission(submission, count, count_actual, is_processed)

      # Skip if already processed
      if is_processed:
        continue
      
      # check if submission title is a question
      if isPotentialChessboardTopic(submission):
        
        # Use CNN to make a prediction
        print("\n---\nImage URL: %s" % submission.url)
        fen, certainty = predictor.makePrediction(submission.url)

        if fen is None:
          print("> %s - Couldn't generate FEN, skipping..." % datetime.now())
          # update & save list
          already_processed.add(submission.id)
          saveProcessed(already_processed)
          addSubmissionToFailures(submission)
          print("\n---\n")
          continue
        
        fen = shortenFEN(fen) # ex. '111pq11r' -> '3pq2r'
        print("Predicted FEN: %s" % fen)
        print("Certainty: %.4f%%" % (certainty*100))

        # Get side from title or fen
        side = getSideToPlay(submission.title, fen)
        # Generate response message
        msg = generateMessage(fen, certainty, side)
        print("fen: %s\nside: %s\n" % (fen, side))

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
            print("\n---\n")
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
    # saveProcessed(already_processed)
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
    print("%s - %d Processed total." % (datetime.now(),len(already_processed)))

print("%s - Program Ended. %d replied / %d read in this session" % (datetime.now(), count_actual, count))
