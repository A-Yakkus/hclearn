#!/bin/bash
echo ${BASH_VERSION}
fname=RunTimer.5000_neurons.log
for I in {1..10}
do
	echo "Iteration "$I":" >> $fname
	/usr/bin/time -va --output=$fname python3 go.py
done
# cat RunTimer.log | grep "wall clock" | tr "walckotimepsdEh()" "\b"| tr ":", "," >> timeData.csv



