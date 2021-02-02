#!/bin/bash

Rzero=2.5
D=14

ns=2000   # num of sequences
nf=16     # num of frames
N=1000    # num of nodes in graph

nc=3      # num of channels, 3 for SIR, 4 for SEIR
nh=128    # hidden dimension
nl=10     # num of gcn layers

dr=0.265  # dropout
lr=0.0033 # learning rate

gt=ER     # random graph type, ER, BA, BA-Tree, RGG
gs=p0.02    # graph specific parameter
	  # ER: p0.02, BA: m10, BA-Tree: m1, RGG: r0.08

bs=16     # batch_size
ep=3      # num of epochs

skip=1
end=-1


prefix=${gt}-SIR-${N}-${Rzero}-${D}
output=${prefix}-sw.dat
if [[ ! -f ${output} ]]; then
    touch ${output}
    echo "Window Acc MRR Hit5 Hit10 Hit20" >> ${output}
fi

for (( ind=0; ind < 5; ind++ )) do
	g="../../dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
	s="../../dataset/delay_SIR/sim/SIR_Rzero${Rzero}_D${D}_N${N}_${gs}_g${ind}.pickle"


	for (( run=0; run < 1; run++ )) do
		python run_pct_slide_T.py --n_node ${N} --batch_size ${bs} --epoch ${ep} --graph $g --seq $s --start ${skip} --end ${end} --n_channel ${nc} --n_hidden ${nh} --n_layer ${nl} --lr ${lr} --dropout ${dr} | tail -n 10 >> ${output}
	done
done




