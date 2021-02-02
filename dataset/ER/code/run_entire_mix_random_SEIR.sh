#!/bin/bash

N=$1      # graph size
nframe=16 # number of frames to sample
ls=2000   # number of sequences, len_seq
gtype=ER  # graph type
stype=$2  # simulation type, SIR, SEIR, etc
gid=$3    # graph id
p=0.02    # random graph probability to connect
f0=0.02   # min outbreak fraction

python sim_entire_mix_random_SEIR.py --n_node ${N} --n_frame ${nframe} --len_seq ${ls} --graph_id ${gid} --graph_type ${gtype} --sim_type ${stype} --p ${p} --f0 ${f0}
