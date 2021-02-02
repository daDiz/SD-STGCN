#!/bin/bash


Rzero=2.5
D=14

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

valid=0
random1=1
random2=0

prefix=${gt}-SIR-${N}-${Rzero}-${D}

for (( ind=0; ind < 5; ind++ )) do
	g="./dataset/${gt}/data/graph/${gt}_N${N}_${gs}_g${ind}.edgelist"
	s="./dataset/delay_SIR/sim/SIR_Rzero${Rzero}_D${D}_N${N}_${gs}_g${ind}.pickle"

	for (( run=0; run < 1; run++ )) do
		python main_SIR_pct_slide_nonMarkovian.py --random ${random1} --valid ${valid} --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --seq $s --start ${skip} --end ${end}

		for w in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
		do
			output=${prefix}-${w}.dat
			if [[ ! -f ${output} ]]; then
				touch ${output}
				echo "Acc MRR Hit5 Hit10 Hit20" >> ${output}
			fi
			python test_SIR_pct_slide_nonMarkovian.py --random ${random2} --valid ${valid} --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --seq $s --start_pct ${w} | tail -n 1 >> ${output}

		done
	done
done




