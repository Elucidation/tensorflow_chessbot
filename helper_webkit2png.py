# Wrapper on webkit2png to render chessboard layouts from lichess

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *
from PyQt4.QtNetwork import *

# Download webkit2png from here and place in same directory
# https://github.com/adamn/python-webkit2png/tree/master/webkit2png
# 
import webkit2png

######################################################3
# For scraping website screenshots
class Options():
  url = None
  output_filename = None
  cookie = None


class ChessScreenshotServer():
  """docstring for ChessScreenshotServer"""
  def __init__(self, url=None, output_filename=None):
    self.options = Options()
    self.options.url = url
    self.options.output_filename = output_filename

    self.app = self.init_qtgui()

  def init_qtgui(self, display=None, style=None, qtargs=None):
    """Initiates the QApplication environment using the given args."""
    if QApplication.instance():
      print ("QApplication has already been instantiated.\n"
         "Ignoring given arguments and returning existing QApplication.")
      return QApplication.instance()

    qtargs2 = [sys.argv[0]]
    qtargs2.extend(qtargs or [])

    return QApplication(qtargs2)

  def renderScreenshotToFile(self):
    """This is run within QT"""
    try:
      renderer = webkit2png.WebkitRenderer()
      # renderer.wait = 5
      renderer.qWebSettings[QWebSettings.JavascriptEnabled] = True # Enable javascript
      if self.options.cookie:
        renderer.cookies = [self.options.cookie]
      with open(self.options.output_filename, 'w') as f:
        renderer.render_to_file(res=self.options.url, file_object=f)
        print "\tSaved screenshot to '%s'" % f.name
      QApplication.exit(0)
    except RuntimeError, e:
      print "Error:", e
      print >> sys.stderr, e
      QApplication.exit(1)


  def takeScreenshot(self, url=None, output_filename=None):
    if url:
      self.options.url = url
    if output_filename:
      self.options.output_filename = output_filename
    
    QTimer.singleShot(0, self.renderScreenshotToFile)
    return self.app.exec_()

  def takeChessScreenshot(self, fen_string=None, output_filename=None, 
                          cookie=None): 
    """Take uncropped screenshot of lichess board of FEN string and save to file"""
    url_template = "http://en.lichess.org/editor/%s"
    if cookie:
      self.options.cookie = cookie
    return self.takeScreenshot(url_template % fen_string, output_filename)