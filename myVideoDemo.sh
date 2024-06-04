#!/usr/bin/env bash
export ASCEND_RT_VISIBLE_DEVICES=2
TORCHRUN_ARGS="--nproc_per_node=1 --nnodes=1 --master_addr=localhost --master_port=12323"
torchrun $TORCHRUN_ARGS myVideoDemo.py --video_path /home/zshuai/TalkwebTestVideoSub --num_queries 700 --pre /home/zshuai/CLTR/weight/nwpu_model_best.pth