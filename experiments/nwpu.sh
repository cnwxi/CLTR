#!/usr/bin/env bash
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 8218 train_distributed.py --gpu_id '0,1' \
# --gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-5 \
# --batch_size 4 --num_patch 1 --threshold 0.35 --num_queries 700 \
# --dataset nwpu --crop_size 256 --pre None --test_per_epoch 20 --test_patch --save
#!/usr/bin/env bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

TORCHRUN_ARGS="--nproc_per_node=4 --nnodes=1 --master_addr=localhost --master_port=12321"

torchrun $TORCHRUN_ARGS train_distributed.py --gpu_id '4,5,6,7' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-5 \
--batch_size 16 --num_patch 1 --threshold 0.35 --num_queries 700 \
--dataset nwpu --crop_size 256 --pre None --test_per_epoch 20 --test_patch --save