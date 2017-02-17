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


def startStream(args):
  reddit = praw.Reddit('CFB') # client credentials set up in local praw.ini file
  cfb = reddit.user.me() # ChessFenBot object
  subreddit = reddit.subreddit('chess+chessbeginners+AnarchyChess+betterchess')
  predictor = tensorflow_chessbot.ChessboardPredictor()

  REPLY_WAIT_TIME = 10 # seconds to wait after successful reply

  # Start live stream on all submissions in the subreddit
  for submission in subreddit.stream.submissions():
    
    # Check if submission passes requirements and wasn't already replied to
    if isPotentialChessboardTopic(submission):
      if not previouslyRepliedTo(submission, cfb):
        # Generate response
        response = generateResponseMessage(submission, predictor)
        if response is None:
          logMessage(submission,"[NO-FEN]") # Skip since couldn't generate FEN
          continue

        # Reply to submission with response
        if not args.dry:
          logMessage(submission,"[REPLIED]")
          submission.reply(response)
        else:
          logMessage(submission,"[DRY-RUN-REPLIED]")

        # Wait after submitting to not overload
        waitWithComments(REPLY_WAIT_TIME)
      else:
        logMessage(submission,"[SKIP]") # Skip since replied to already

    else:
      logMessage(submission)
      time.sleep(1) # Wait a second between normal submissions


def dryRunTest(submission='5tuerh'):
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


def main(args):
  running = True
  while running:
    try:
      startStream(args)
    except (socket.error, requests.exceptions.ReadTimeout,
            requests.packages.urllib3.exceptions.ReadTimeoutError,
            requests.exceptions.ConnectionError) as e:
      print(
        "> %s - Connection error, retrying in 30 seconds: %s" % (
        datetime.now(), e))
      time.sleep(30)
      continue
    except Exception as e:
      print("Unknown Error, attempting restart in 30 seconds:",e)
      time.sleep(30)
      continue
    except KeyboardInterrupt:
      print("Keyboard Interrupt: Exiting...")
      running = False
  print('Finished')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dry', help='dry run (don\'t actually submit replies)',
                      action="store_true", default=False)
  parser.add_argument('--test', help='Dry run test on pre-existing comment)',
                      action="store_true", default=False)
  args = parser.parse_args()
  if args.test:
    print('Doing dry run test on submission')
    dryRunTest()
  else:
    main(args)
