#!/bin/bash

N=$1      # graph size
nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=ER  # graph type
stype=$2  # simulation type, SIR, SEIR, etc
R0=$3     # R0
gamma=$4  # gamma
#gid=$5    # graph id
alpha=0.5 # alpha
p=0.02    # random graph probability to connect
f0=0.02   # min outbreak fraction

for (( gid=0; gid < 5; gid++ )) do
	nohup python sim_entire.py --n_node ${N} --n_frame ${nframe} --len_seq ${ls} --graph_id ${gid} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --gamma ${gamma} --alpha ${alpha} --p ${p} --f0 ${f0} > ./simulation_length/${stype}_${N}-${R0}-${gamma}-${gid}-entire.dat & 
done

