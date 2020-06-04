#! /bin/bash
DIR=$1_$(date +"%Y_%m_%d_%H_%M_%S")
mkdir $DIR
python -m cProfile -o $DIR/profile.prof ../src/main/go.py
python statstocsv.py $DIR/profile.prof $DIR/profile_csv.csv
#python displayStats.py
