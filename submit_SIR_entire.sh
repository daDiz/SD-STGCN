#!/bin/bash

ind=$1

Rzero1=2.5 # train simulation R0
gamma1=0.4 # train simulation gamma

Rzero2=2.5
gamma2=0.4


ns=2000   # num of sequences
nf=16     # num of frames
N=1000    # num of nodes in graph

gt=ER     # random graph type, ER, BA, BA-Tree, RGG
gs=p0.02    # graph specific parameter
	  # ER: p0.02, BA: m10, BA-Tree: m1, RGG: r0.08

bs=16     # batch_size
ep=3      # num of epochs
ks=4      # spatio kernel size
kt=3      # temporal kernel size
sc=gcn    # spatio convolution layer type
save=1    # save every # of epochs

skip=1
end=-1

valid=0  # if evaluate performance on validation set during training
random=1


g="./dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
s1="./dataset/${gt}/data/SIR/SIR_Rzero${Rzero1}_gamma${gamma1}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"
s2="./dataset/${gt}/data/SIR/SIR_Rzero${Rzero2}_gamma${gamma2}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"

python main_SIR_entire.py --valid ${valid} --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --seq $s1 --start ${skip} --end ${end}

python test_SIR_entire.py --valid ${valid} --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --seq $s2 --start ${skip} --end ${end}


