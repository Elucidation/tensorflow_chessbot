**TL;DR** : [TensorFlow](www.tensorflow.org) [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) algorithm trained on 32x32 grayscale chess tiles predicts chessboards layouts from online chessboard screenshots, 

[/u/ChessFenBot](https://www.reddit.com/user/ChessFenBot) will automatically reply to [reddit /r/chess](https://www.reddit.com/r/) image posts with online chessboard screenshots with a predicted [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and [lichess](http://www.lichess.org) analysis link.

# Goal
Build a Reddit bot ([/u/ChessFenBot](https://www.reddit.com/user/ChessFenBot) is up) that listens on /r/chess and replies to posts with screenshots of chessboards with a link to the lichess.org analysis page and the FEN ([Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)) string of the board.

```py
predictor = ChessboardPredictor()
fen, fen_img_link, certainty = predictor.makePrediction('http://imgur.com/u4zF5Hj.png')
print "Predicted FEN: %s" % fen
print "Certainty: %.1f%%" % (certainty*100)
```

```
Setting up CNN TensorFlow graph...
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 8
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 8
Loading model 'saved_models/model_10000.ckpt'
Model restored.
Certainty range [0.999545 - 1], Avg: 0.999977, Overall: 0.998546
Predicted FEN: 8/5p2/5k1P/2p4P/1p1p4/8/3K4/8
Certainty: 99.9%
Done
[Finished in 1.8s]
```

### Reddit Bot Comment Example

ChessFenBot automatically replied to [this reddit post](https://www.reddit.com/r/chess/comments/45osos/very_difficult_find_the_best_move_for_white/d004cg6?context=3), it processed the [screenshot link url](http://i.imgur.com/HnWYt8A.png) and responded with:

> ChessFenBot [◕ _ ◕]<sup>\* *I make FENs*</sup>
> 
> ---
> 
> I attempted to generate a chessboard layout from the posted image, with an overall certainty of **99.9916%**.
> 
> FEN: [1nkr4/1p3q1p/pP4pn/P1r5/3N1p2/2b2B1P/5PPB/2RQ1RK1](http://www.fen-to-image.com/image/30/1nkr1111/1p111q1p/pP1111pn/P1r11111/111N1p11/11b11B1P/11111PPB/11RQ1RK1.png)
> 
> Here is a link to a [Lichess Analysis](http://www.lichess.org/analysis/1nkr4/1p3q1p/pP4pn/P1r5/3N1p2/2b2B1P/5PPB/2RQ1RK1_w) - White to play
> 
> ---
> 
> <sup>Yes I am a machine learning bot | [`How I work`](https://github.com/Elucidation/tensorflow_chessbot 'Must go deeper') | Reply with a corrected FEN or [Editor link)](http://www.lichess.org/editor/r1b1r1k1/5pp1/p1pR1nNp/8/2B5/2q5/P1P1Q1PP/5R1K) to add to my next training dataset</sup>

## Workflow

There are three ipython notebooks which show the workflow from turning a screenshot of a chessboard into a set of 32x32 grayscale tiles, to generating those tiles for training and testing, and then the actual training and learning of the neural network from those trials using [TensorFlow](http://www.tensorflow.org).

1. [tensorflow_chessbot.ipynb](tensorflow_chessbot.ipynb) - Computer Vision
1. [tensorflow_generate_training_data.ipynb](tensorflow_generate_training_data.ipynb) - Generating a dataset from set of screenshots of chessboards in known configurations
1. [tensorflow_learn.ipynb](tensorflow_learn.ipynb) - **TensorFlow Neural Network Training & Prediction** Basic Regression classifier, works for more common lichess.org and chess.com screenshots
1. [tensorflow_learn_cnn.ipynb](tensorflow_learn_cnn.ipynb) - **TensorFlow Convolutional Neural Network Training & Prediction** tested with ~73% success rate on 71 chess subreddit posts

### #1 [tensorflow_chessbot.ipynb](tensorflow_chessbot.ipynb) - Computer Vision

Here is a screenshot with the detected lines of the chessboard overlaid, showing where we'll cut the image into tiles.

![overlay lines](readme_images/overlay_lines.png)

### #2 [tensorflow_generate_training_data.ipynb](tensorflow_generate_training_data.ipynb) - Generating a dataset

[Lichess.org](lichess.org) provides a URL interface with a FEN string that loads a webpage with that board arrayed. A nice repo called [pythonwebkit2png](https://github.com/adamn/python-webkit2png) provides a way to render webpages programmatically, allowing us to generate several (80 in thise) random FEN strings, load the URL and take a screenshot all automatically.

![random fen](readme_images/random_fen.png)

Here is 5 example tiles and their associated label, a 13 length one-hot vector corresponding to 6 white pieces, 6 black pieces, and 1 empty space.

![dataset example](readme_images/dataset_example.png)


### #3 [tensorflow_learn.ipynb](tensorflow_learn.ipynb) - **TensorFlow Neural Network Training & Prediction**

We train the neural network on generated data from 80 lichess.org screenshots, which is 5120 tiles. We test it with 5 screenshots (320 tiles) as a quick sanity check. Here is a visualization of the weights for the white King, Queen and Rook.

![Some weights](readme_images/weight_KQR.png)

Finally we can make predictions on images passed by URL, the ones from lichess and visually similar boards work well, the ones that are too different from what we trained for don't work, suggesting that getting more data is in order. Here is a prediction on the image for [this reddit post](https://www.reddit.com/r/chess/comments/45inab/moderate_black_to_play_and_win/)

![Prediction](readme_images/prediction.png)

### #4 [tensorflow_learn_cnn.ipynb](tensorflow_learn_cnn.ipynb) - **TensorFlow Convolutional Neural Network Training & Prediction**

Built a slightly larger dataset of ~150 screenshots which is around 9600 tiles which includes randomized FEN diagrams from lichess.org, chess.com, and 2 FEN generated diagram sites.

Tested with ~73% success rate on 71 chess subreddit posts, good enough to make a first draft Reddit bot.


### Ideation
Reddit post has an image link (perhaps as well as a statement "white/black to play").

Bot takes the image, uses some CV to find a chessboard on it, splits up into
a set of images of squares. These are the inputs to the tensorflow CNN
which will return probability of which piece is on it (or empty)

Dataset will include chessboard squares from chess.com, lichess
Different styles of each, all the pieces

Generate synthetic data via added noise:
 * change in coloration
 * highlighting
 * occlusion from lines etc.

Take most probable set from TF response, use that to generate a FEN of the
board, and bot comments on thread with FEN and link to lichess analysis
