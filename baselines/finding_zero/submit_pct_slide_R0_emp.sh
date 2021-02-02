#!/bin/bash

Rzero=2.5
D=14

ns=2000   # num of sequences
nf=16     # num of frames

N=58    # num of nodes bkFratB
#N=774   # num of nodes highSchool
#N=403   # num of nodes sfhh

nc=3      # num of channels, 3 for SIR, 4 for SEIR
nh=128    # hidden dimension
nl=10     # num of gcn layers

dr=0.265  # dropout
lr=0.0033 # learning rate

gt=bkFratB     # random graph type, ER, BA, BA-Tree, RGG
#gt=highSchool
#gt=sfhh

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
	g="../../dataset/${gt}/data/graph/${gt}.edgelist"
	s="../../dataset/delay_SIR/sim/SIR_Rzero${Rzero}_D${D}_${gt}.pickle"


	for (( run=0; run < 1; run++ )) do
		python run_pct_slide_T_emp.py --n_node ${N} --batch_size ${bs} --epoch ${ep} --graph $g --seq $s --start ${skip} --end ${end} --n_channel ${nc} --n_hidden ${nh} --n_layer ${nl} --lr ${lr} --dropout ${dr} | tail -n 13 >> ${output}
	done
done




