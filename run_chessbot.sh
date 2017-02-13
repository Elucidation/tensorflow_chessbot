#!/bin/bash 
echo "Currently in `pwd`"
python -u ./chessbot.py > ./out.log 2> ./out_error.log
