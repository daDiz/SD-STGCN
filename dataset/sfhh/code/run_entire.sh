#!/bin/bash

N=403      # graph size
nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=sfhh  # graph type
stype=SIR  # simulation type, SIR, SEIR, etc
R0=$1     # R0
gamma=$2  # gamma
f0=0.02   # min outbreak fraction

python sim_entire.py --n_node ${N} --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --gamma ${gamma} --f0 ${f0}
