#!/bin/bash

N=$1      # graph size
nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=BA  # graph type
stype=$2  # simulation type, SIR, SEIR, etc
R0=$3     # R0
gamma=$4  # gamma
gid=$5    # graph id
alpha=0.5 # alpha
m=10    
f0=0.02   # min outbreak fraction
T=30      # simulation uplimit

python sim_T.py --T ${T} --n_node ${N} --n_frame ${nframe} --len_seq ${ls} --graph_id ${gid} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --gamma ${gamma} --alpha ${alpha} --m ${m} --f0 ${f0}
