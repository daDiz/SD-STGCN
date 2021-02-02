#!/bin/bash

ind=$1

Rzero=2.5 # simulation R0
gamma=0.4 # simulation gamma
ns=2000   # num of sequences
N=1000    # num of nodes in graph
nf=16     # num of frame, for the sequence file only

nc=3      # num of channels, 3 for SIR, 4 for SEIR
nh=128    # hidden dimension
nl=10     # num of gcn layers

dr=0.265  # dropout
lr=0.0033 # learning rate

gt=ER     # random graph type, ER, BA, BA-Tree, RGG
gs=p0.02  # graph specific parameter
	  # ER: p0.02, BA: m10, BA-Tree: m1, RGG: r0.08

bs=16     # batch_size
ep=10     # num of epochs

skip=1
end=-1

T=30


g="../../dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
s="../../dataset/${gt}/data/SIR/SIR_Rzero${Rzero}_gamma${gamma}_T${T}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"

python run_T.py --n_node ${N} --batch_size ${bs} --epoch ${ep} --graph $g --seq $s --start ${skip} --end ${end} --n_channel ${nc} --n_hidden ${nh} --n_layer ${nl} --lr ${lr} --dropout ${dr} 

