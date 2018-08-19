#!/usr/bin/env python3
#
# usage: tileset_generator.py [-h] input_folder output_folder

# Generate tile images for alll chessboard images in input folder

# positional arguments:
#   input_folder   Input image folder
#   output_folder  Output tile folder

# optional arguments:
#   -h, --help     show this help message and exit

# Pass an input folder and output folder
# Builds tile images for each chessboard image in input folder and puts
# in the output folder
# Used for building training datasets
from chessboard_finder import *
import os
import glob

def saveTiles(tiles, img_save_dir, img_file):
  letters = 'ABCDEFGH'
  if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
  
  for i in range(64):
    sqr_filename = "%s/%s_%s%d.png" % (img_save_dir, img_file, letters[i%8], i/8+1)
    
    # Make resized 32x32 image from matrix and save
    if tiles.shape != (32,32,64):
      PIL.Image.fromarray(tiles[:,:,i]) \
          .resize([32,32], PIL.Image.ADAPTIVE) \
          .save(sqr_filename)
    else:
      # Possibly saving floats 0-1 needs to change fromarray settings
      PIL.Image.fromarray((tiles[:,:,i]*255).astype(np.uint8)) \
          .save(sqr_filename)

def generateTileset(input_chessboard_folder, output_tile_folder):
  # Create output folder as needed
  if not os.path.exists(output_tile_folder):
    os.makedirs(output_tile_folder)

  # Get all image files of type png/jpg/gif
  img_files = set(glob.glob("%s/*.png" % input_chessboard_folder))\
    .union(set(glob.glob("%s/*.jpg" % input_chessboard_folder)))\
    .union(set(glob.glob("%s/*.gif" % input_chessboard_folder)))

  num_success = 0
  num_failed = 0
  num_skipped = 0

  for i, img_path in enumerate(img_files):
    print("#% 3d/%d : %s" % (i+1, len(img_files), img_path))
    # Strip to just filename
    img_file = img_path[len(input_chessboard_folder):-4]

    # Create output save directory or skip this image if it exists
    img_save_dir = "%s/tiles_%s" % (output_tile_folder, img_file)
    
    if os.path.exists(img_save_dir):
      print("\tSkipping existing")
      num_skipped += 1
      continue
    
    # Load image
    print("---")
    print("Loading %s..." % img_path)
    img_arr = np.array(loadImageGrayscale(img_path), dtype=np.float32)

    # Get tiles
    print("\tGenerating tiles for %s..." % img_file)
    corners = findChessboardCorners(img_arr)
    tiles = getChessTilesGray(img_arr, corners)

    # Save tiles
    if len(tiles) > 0:
      print("\tSaving tiles %s" % img_file)
      saveTiles(tiles, img_save_dir, img_file)
      num_success += 1
    else:
      print("\tNo Match, skipping")
      num_failed += 1

  print("\t%d/%d generated, %d failures, %d skipped." % (num_success,
    len(img_files) - num_skipped, num_failed, num_skipped))

if __name__ == '__main__':
  np.set_printoptions(suppress=True, precision=2)
  parser = argparse.ArgumentParser(description='Generate tile images for alll chessboard images in input folder')
  parser.add_argument('input_folder', metavar='input_folder', type=str,
                      help='Input image folder')
  parser.add_argument('output_folder', metavar='output_folder', type=str,
                      help='Output tile folder')
  args = parser.parse_args()
  generateTileset(args.input_folder, args.output_folder)