# -*- coding: utf-8 -*-
# Response message template
MESSAGE_TEMPLATE = """[◕ _ ◕]^*

I attempted to generate a [chessboard layout]({unaligned_fen_img_link}) from the posted image[^(what I saw)]({visualize_link}),
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
^( | )[^(`Try your own images`)](http://tetration.xyz/ChessboardFenTensorflowJs/)
^( | Reply with a corrected FEN to add to my next training dataset)

"""