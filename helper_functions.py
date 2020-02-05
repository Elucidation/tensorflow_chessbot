import numpy as np

# Imports for visualization
import PIL.Image

# DEBUG for ipython notebook visualizations.
# from IPython.display import clear_output, Image, display

# def display_array(a, fmt='jpeg', rng=[0,1]):
#   """Display an array as a picture."""
#   a = (a - rng[0])/float(rng[1] - rng[0]) # normalized float value
#   a = np.uint8(np.clip(a*255, 0, 255))
#   f = StringIO()

#   PIL.Image.fromarray(np.asarray(a, dtype=np.uint8)).save(f, fmt)
#   display(Image(data=f.getvalue()))

# def display_weight(a, fmt='jpeg', rng=[0,1]):
#   """Display an array as a color picture."""
#   a = (a - rng[0])/float(rng[1] - rng[0]) # normalized float value
#   a = np.uint8(np.clip(a*255, 0, 255))
#   f = StringIO()

#   v = np.asarray(a, dtype=np.uint8)

#   # blue is high intensity, red is low
#   # Negative
#   r = 255-v.copy()
#   r[r<127] = 0
#   r[r>=127] = 255
  
#   # None
#   g = np.zeros_like(v)
  
#   # Positive
#   b = v.copy()
#   b[b<127] = 0
#   b[b>=127] = 255
  
#   #np.clip((v-127)/2,0,127)*2

#   #-1 to 1
#   intensity = np.abs(2.*a-1)

#   rgb = np.uint8(np.dstack([r,g,b]*intensity))

#   PIL.Image.fromarray(rgb).save(f, fmt)
#   display(Image(data=f.getvalue(), width=100))

# def display_image(a, fmt='png'):
#   """Display an image as a picture in-line."""
#   f = StringIO()

#   PIL.Image.fromarray(np.asarray(a, dtype=np.uint8)).save(f, fmt)
#   display(Image(data=f.getvalue()))

# FEN related
def getFENtileLabel(fen,letter,number):
  """Given a fen string and a rank (number) and file (letter), return label vector"""
  l2i = lambda l:  ord(l)-ord('A') # letter to index
  number = 8-number # FEN has order backwards
  piece_letter = fen[number*8+number + l2i(letter)]
  label = np.zeros(13, dtype=np.uint8)
  label['1KQRBNPkqrbnp'.find(piece_letter)] = 1 # note the 1 instead of ' ' due to FEN notation
  # We ignore shorter FENs with numbers > 1 because we generate the FENs ourselves
  return label

# We'll define the 12 pieces and 1 spacewith single characters 
#  KQRBNPkqrbnp
def getLabelForSquare(letter,number):
  """Given letter and number (say 'B3'), return one-hot label vector
     (12 pieces + 1 space == no piece, so 13-long vector)"""
  l2i = lambda l:  ord(l)-ord('A') # letter to index
  piece2Label = lambda piece: ' KQRBNPkqrbnp'.find(piece)
  # build mapping to index
  # Starter position
  starter_mapping = np.zeros([8,8], dtype=np.uint8)
  starter_mapping[0, [l2i('A'), l2i('H')]] = piece2Label('R')
  starter_mapping[0, [l2i('B'), l2i('G')]] = piece2Label('N')
  starter_mapping[0, [l2i('C'), l2i('F')]] = piece2Label('B')
  starter_mapping[0, l2i('D')] = piece2Label('Q')
  starter_mapping[0, l2i('E')] = piece2Label('K')
  starter_mapping[1, :] = piece2Label('P')

  starter_mapping[7, [l2i('A'), l2i('H')]] = piece2Label('r')
  starter_mapping[7, [l2i('B'), l2i('G')]] = piece2Label('n')
  starter_mapping[7, [l2i('C'), l2i('F')]] = piece2Label('b')
  starter_mapping[7, l2i('D')] = piece2Label('q')
  starter_mapping[7, l2i('E')] = piece2Label('k')
  starter_mapping[6, :] = piece2Label('p')
  # Note: if we display the array, the first row is white,
  # normally bottom, but arrays show it as top
  
  # Generate one-hot label
  label = np.zeros(13, dtype=np.uint8)
  label[starter_mapping[number-1, l2i(letter), ]] = 1
  return label

def name2Label(name):
  """Convert label vector into name of piece"""
  return ' KQRBNPkqrbnp'.find(name)

def labelIndex2Name(label_index):
  """Convert label index into name of piece"""
  return ' KQRBNPkqrbnp'[label_index]

def label2Name(label):
  """Convert label vector into name of piece"""
  return labelIndex2Name(label.argmax())

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

def unflipFEN(fen):
    if len(fen) < 71:
        fen = lengthenFEN(FEN)
    return '/'.join([ r[::-1] for r in fen.split('/') ][::-1])


# For Training in IPython Notebooks
def loadFENtiles(image_filepaths):
  """Load Tiles with FEN string in filename for labels.
  return both images and labels"""
  # Each tile is a 32x32 grayscale image, add extra axis for working with MNIST Data format
  images = np.zeros([image_filepaths.size, 32, 32, 1], dtype=np.uint8)
  labels = np.zeros([image_filepaths.size, 13], dtype=np.float64)

  for i, image_filepath in enumerate(image_filepaths):
    if i % 1000 == 0:
      #print("On #%d/%d : %s" % (i,image_filepaths.size, image_filepath))
      print(".",)
    
    # Image
    images[i,:,:,0] = np.asarray(PIL.Image.open(image_filepath), dtype=np.uint8)

    # Label
    fen = image_filepath[-78:-7]
    _rank = image_filepath[-6]
    _file = int(image_filepath[-5])
    labels[i,:] = getFENtileLabel(fen, _rank, _file)
  print("Done")
  return images, labels

def loadLabels(image_filepaths):
  """Load label vectors from list of image filepaths"""
  # Each filepath contains which square we're looking at, 
  # since we're in starter position, we know which
  # square has which piece, 12 distinct pieces 
  # (6 white and 6 black) and 1 as empty = 13 labels
  training_data = np.zeros([image_filepaths.size, 13], dtype=np.float64)
  for i, image_filepath in enumerate(image_filepaths):
    training_data[i,:] = getLabelForSquare(image_filepath[-6],int(image_filepath[-5]))
  return training_data

def loadImages(image_filepaths):
  # Each tile is a 32x32 grayscale image, add extra axis for working with MNIST Data format
  training_data = np.zeros([image_filepaths.size, 32, 32, 1], dtype=np.uint8)
  for i, image_filepath in enumerate(image_filepaths):
    if i % 100 == 0:
      print("On #%d/%d : %s" % (i,image_filepaths.size, image_filepath))
    img = PIL.Image.open(image_filepath)
    training_data[i,:,:,0] = np.asarray(img, dtype=np.uint8)
  return training_data
