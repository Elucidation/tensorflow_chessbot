#!/usr/bin/env python
# ChessFenBot daemon
# Finds submissions with chessboard images in them,
# use a tensorflow convolutional neural network to predict pieces and return
# a lichess analysis link and FEN diagram of chessboard
# Run with --dry to dry run without actual submissions
from __future__ import print_function
import praw
import requests
import socket
import time
from datetime import datetime
import argparse

import tensorflow_chessbot # For neural network model
from helper_functions_chessbot import *
from helper_functions import shortenFEN
from cfb_helpers import * # logging, comment waiting and self-reply helpers

def generateResponseMessage(submission, predictor):
  print("\n---\nImage URL: %s" % submission.url)
  
  # Use CNN to make a prediction
  fen, certainty, visualize_link = predictor.makePrediction(submission.url)

  if fen is None:
    print("> %s - Couldn't generate FEN, skipping..." % datetime.now())
    print("\n---\n")
    return None
  
  fen = shortenFEN(fen) # ex. '111pq11r' -> '3pq2r'
  print("Predicted FEN: %s" % fen)
  print("Certainty: %.4f%%" % (certainty*100))

  # Get side from title or fen
  side = getSideToPlay(submission.title, fen)
  # Generate response message
  msg = generateMessage(fen, certainty, side, visualize_link)
  print("fen: %s\nside: %s\n" % (fen, side))
  return msg


def processSubmission(submission, cfb, predictor, args, reply_wait_time=10):
  # Check if submission passes requirements and wasn't already replied to
  if isPotentialChessboardTopic(submission):
    if not previouslyRepliedTo(submission, cfb):
      # Generate response
      response = generateResponseMessage(submission, predictor)
      if response is None:
        logMessage(submission,"[NO-FEN]") # Skip since couldn't generate FEN
        return

      # Reply to submission with response
      if not args.dry:
        logMessage(submission,"[REPLIED]")
        submission.reply(response)
      else:
        logMessage(submission,"[DRY-RUN-REPLIED]")

      # Wait after submitting to not overload
      waitWithComments(reply_wait_time)
    else:
      logMessage(submission,"[SKIP]") # Skip since replied to already

  else:
    logMessage(submission)
    time.sleep(1) # Wait a second between normal submissions

def main(args):
  resetTensorflowGraph()
  running = True
  reddit = praw.Reddit('CFB') # client credentials set up in local praw.ini file
  cfb = reddit.user.me() # ChessFenBot object
  subreddit = reddit.subreddit('chess+chessbeginners+AnarchyChess+betterchess+chesspuzzles')
  predictor = tensorflow_chessbot.ChessboardPredictor()

  while running:
    # Start live stream on all submissions in the subreddit
    stream = subreddit.stream.submissions()
    try:
      for submission in stream:
        processSubmission(submission, cfb, predictor, args)
    except (socket.error, requests.exceptions.ReadTimeout,
            requests.packages.urllib3.exceptions.ReadTimeoutError,
            requests.exceptions.ConnectionError) as e:
      print(
        "> %s - Connection error, skipping and continuing in 30 seconds: %s" % (
        datetime.now(), e))
      time.sleep(30)
      continue
    except Exception as e:
      print("Unknown Error, skipping and continuing in 30 seconds:",e)
      time.sleep(30)
      continue
    except KeyboardInterrupt:
      print("Keyboard Interrupt: Exiting...")
      running = False
      break

  predictor.close()
  print('Finished')

def resetTensorflowGraph():
  """WIP needed to restart predictor after an error"""
  import tensorflow as tf
  print('Reset TF graph')
  tf.reset_default_graph() # clear out graph

def runSpecificSubmission(args):
  resetTensorflowGraph()
  reddit = praw.Reddit('CFB') # client credentials set up in local praw.ini file
  cfb = reddit.user.me() # ChessFenBot object
  predictor = tensorflow_chessbot.ChessboardPredictor()

  submission = reddit.submission(args.sub)
  print("URL: ", submission.url)
  if submission:
    print('Processing...')
    processSubmission(submission, cfb, predictor, args)

  predictor.close()
  print('Done')

def dryRunTest(submission='5tuerh'):
  resetTensorflowGraph()
  reddit = praw.Reddit('CFB') # client credentials set up in local praw.ini file
  predictor = tensorflow_chessbot.ChessboardPredictor()

  # Use a specific submission
  submission = reddit.submission(submission)
  print('Loading %s' % submission.id)
  # Check if submission passes requirements and wasn't already replied to
  if isPotentialChessboardTopic(submission):
    # Generate response
    response = generateResponseMessage(submission, predictor)
    print("RESPONSE:\n")
    print('-----------------------------')
    print(response)
    print('-----------------------------')
  else:
    print('Submission not considered chessboard topic')

  predictor.close()
  print('Finished')

  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dry', help='dry run (don\'t actually submit replies)',
                      action="store_true", default=False)
  parser.add_argument('--test', help='Dry run test on pre-existing comment)',
                      action="store_true", default=False)
  parser.add_argument('--sub', help='Pass submission string to process')
  args = parser.parse_args()
  if args.test:
    print('Doing dry run test on submission')
    if args.sub:
      dryRunTest(args.sub)
    else:
      dryRunTest()
  elif args.sub is not None:
    runSpecificSubmission(args)
  else:
    main(args)
