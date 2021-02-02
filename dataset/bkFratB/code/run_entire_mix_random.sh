#!/bin/bash

nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=bkFratB  # graph type
stype=SIR  # simulation type, SIR, SEIR, etc
f0=0.1   # min outbreak fraction

python sim_entire_mix_random.py --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --f0 ${f0}
