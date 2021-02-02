#!/bin/bash

ind=$1

Rzero1=1-15 # simulation R0
gamma1=0.1-0.4 # simulation gamma
alpha1=0.2-1.0

Rzero2=2.5
gamma2=0.4
alpha2=0.5


ns=2000   # num of sequences
N=1000    # num of nodes in graph
nf=16     # num of frame, for the sequence file only

nc=4      # num of channels, 3 for SIR, 4 for SEIR
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

save='./output/model/SEIR/exp1/finding_zero'

g="../../dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
s1="../../dataset/${gt}/data/SEIR/SEIR_Rzero${Rzero1}_gamma${gamma1}_alpha${alpha1}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"
s2="../../dataset/${gt}/data/SEIR/SEIR_Rzero${Rzero2}_gamma${gamma2}_alpha${alpha2}_ls${ns}_nf${nf}_N${N}_${gs}_g${ind}_entire.pickle"
	
output=${gt}-SEIR-${N}-${Rzero1}-${gamma1}-${alpha1}-${Rzero2}-${gamma2}-${alpha2}.dat

python run_SEIR.py --n_node ${N} --batch_size ${bs} --epoch ${ep} --graph $g --seq_train ${s1} --start ${skip} --end ${end} --n_channel ${nc} --n_hidden ${nh} --n_layer ${nl} --lr ${lr} --dropout ${dr} --save ${save} --mode train 

python run_SEIR.py --n_node ${N} --batch_size ${bs} --epoch ${ep} --graph $g --seq_test ${s2} --start ${skip} --end ${end} --n_channel ${nc} --n_hidden ${nh} --n_layer ${nl} --lr ${lr} --dropout ${dr} --save ${save} --mode test 

