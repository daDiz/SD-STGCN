#!/bin/bash

NN=100 # num of nodes to calculate at once
gid=0
gt=ER # ER, BA, BA-Tree, RGG
for T in 1 5 10 15 20 25
do
	for (( i=0; i < 10; i++ ))
	do
		(( start = i * NN ))
		(( end = (i+1) * NN ))
		nohup python dmp_pred.py ${gid} ${T} ${start} ${end} ${gt} >> dmp_pred_${T}.log &
	done
done
