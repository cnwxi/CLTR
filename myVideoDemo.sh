#!/usr/bin/env bash
export ASCEND_RT_VISIBLE_DEVICES=0
TORCHRUN_ARGS="--nproc_per_node=1 --nnodes=1 --master_addr=localhost --master_port=12322"
torchrun $TORCHRUN_ARGS myVideoDemo.py --video_path /home/zshuai/TalkwebTestVideo --num_queries 700 --pre /home/zshuai/CLTR/save_file/log_file/20240520_1029/model_best.pth