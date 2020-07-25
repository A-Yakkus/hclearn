#!/bin/bash
echo ${BASH_VERSION}
fname=RunTimer.1086_neurons.annotated.1
logext=.log
csvext=.csv
for I in {1..5}
do
	echo "Iteration "$I":" >> $fname$logext
	/usr/bin/time -va --output=$fname$logext python3 go.py
done
cat $fname$logext | grep "wall clock" | tr "walckotimepsdEh()" "\b"| tr ":", "," >> $fname$csvext



