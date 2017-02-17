import numpy as np

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
import urllib2

# Imports for pulling metadata from imgur url
import requests
from bs4 import BeautifulSoup

def loadImage(img_file):
  """Load image from file, convert to grayscale float32 numpy array"""
  img = PIL.Image.open(img_file)

  # Convert to grayscale and return as an numpy array
  return np.asarray(img.convert("L"), dtype=np.float32)

def loadImageFromURL(url):
  """Load image from url.
  Or metadata url link from imgur"""
  
  # If imgur try to load from metadata
  if 'imgur' in url:
    url = tryUpdateImgurURL(url)

  # Otherwise try loading image from url directly
  try:
    req = urllib2.Request(url, headers={'User-Agent' : "TensorFlow Chessbot"})
    con = urllib2.urlopen(req)
    return PIL.Image.open(StringIO(con.read())), url
  except IOError, e:
    return None, url

def tryUpdateImgurURL(url):
  """Try to get actual image url from imgur metadata"""
  soup = BeautifulSoup(requests.get(url).content, "lxml")
  
  # Get metadata tags
  meta = soup.find_all('meta')
  # Get the specific tag, ex.
  # <meta content="https://i.imgur.com/bStt0Fuh.jpg" name="twitter:image"/>
  tags = list(filter(lambda tag: 'name' in tag.attrs and tag.attrs['name'] == "twitter:image", meta))
  
  if tags:
    # Replace url with metadata url
    url = tags[0]['content']
  
  return url

def loadImageFromPath(img_path):
  """Load PIL image from image filepath, keep as color"""
  return PIL.Image.open(open(img_path,'rb'))


def resizeAsNeeded(img):
  """Resize if image larger than 2k pixels on a side"""
  if img.size[0] > 2000 or img.size[1] > 2000:
    print("Image too big (%d x %d)" % (img.size[0], img.size[1]))
    new_size = 500.0 # px
    if img.size[0] > img.size[1]:
      # resize by width to new limit
      ratio = new_size / img.size[0]
    else:
      # resize by height
      ratio = new_size / img.size[1]
    print("Reducing by factor of %.2g" % (1./ratio))
    img = img.resize(img.size * ratio, PIL.Image.ADAPTIVE)
    print("New size: (%d x %d)" % (img.size[0], img.size[1]))
  return img

def getVisualizeLink(corners, url):
  """Return online link to visualize found corners for url"""
  encoded_url = urllib2.quote(url, safe='')
  
  return ("http://tetration.xyz/tensorflow_chessbot/overlay_chessboard.html?%d,%d,%d,%d,%s" % 
    (corners[0], corners[1], corners[2], corners[3], encoded_url))