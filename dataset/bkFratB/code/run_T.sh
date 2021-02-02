#!/bin/bash

nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=bkFratB  # graph type
stype=SIR  # simulation type, SIR, SEIR, etc
R0=2.5     # R0
gamma=0.2  # gamma
f0=0.1    # min outbreak fraction
T=30      # simulation uplimit

python sim_T.py --T ${T} --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --gamma ${gamma} --f0 ${f0}
