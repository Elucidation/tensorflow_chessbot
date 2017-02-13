TensorFlow Chessbot - /u/ChessFenBot [◕ _ ◕]<sup>\* *I make FENs*</sup>
---
### Reddit Bot

[/u/ChessFenBot](https://www.reddit.com/user/ChessFenBot) will automatically reply to [reddit /r/chess](https://www.reddit.com/r/) new topic image posts that contain detectable online chessboard screenshots. A screenshot either ends in `.png`, `.jpg`, `.gif`, or is an `imgur` link. 

It replies with a [lichess](http://www.lichess.org) analysis link for that layout and a predicted [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).

```py
predictor = ChessboardPredictor()
fen, certainty = predictor.makePrediction('http://imgur.com/u4zF5Hj.png')
print "Predicted FEN: %s" % fen
print "Certainty: %.1f%%" % (certainty*100)
```

```
Certainty range [0.999545 - 1], Avg: 0.999977, Overall: 0.998546
Predicted FEN: 8/5p2/5k1P/2p4P/1p1p4/8/3K4/8
Certainty: 99.9%
Done
[Finished in 1.8s]
```

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

## Running

Automated build on Docker available at `elucidation/tensorflow_chessbot`

Using your own `auth_config.py` which has the form

```py
USERNAME='<NAME>'
PASSWORD='<PASSWORD>'
USER_AGENT='<AGENT INFO>'
```

You can download and run the docker image using:

```
docker run -dt --rm --name cfb -v <local_auth_file>:/tcb/auth_config.py elucidation/tensorflow_chessbot
```