#!/bin/bash


ind=$1    # graph index

Rzero=2.5 # simulation R0
gamma=0.4 # simulation gamma
ns=2000   # num of sequences
nf=16     # num of frames
N=1000    # num of nodes in graph

gt=ER     # random graph type, ER, BA, BA-Tree, RGG
gs=p0.02  # graph specific parameter
	  # ER: p0.02, BA: m10, BA-Tree: m1, RGG: r0.08

bs=16     # batch_size
ep=3      # num of epochs
ks=4      # spatio kernel size
kt=3      # temporal kernel size
sc=gcn    # spatio convolution layer type
save=1    # save every # of epochs

skip=1
end=-1

T=30

g="./dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
p="./output/models/${gt}/pred_${gt}_N${N}_${gs}_nf${nf}_g${ind}.pickle"
s="./dataset/${gt}/data/SIR/SIR_Rzero${Rzero}_gamma${gamma}_T${T}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"

python main_SIR_T.py --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --pred $p --seq $s --start ${skip} --end ${end}

