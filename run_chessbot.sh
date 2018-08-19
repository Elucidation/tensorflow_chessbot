#!/bin/bash 
echo "Currently in `pwd`"
log_name=`date +"%F_%H-%M-%S"`
python -u ./chessbot.py > ./out_$log_name.log 2> ./error_$log_name.log
