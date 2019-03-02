#!/bin/bash


base_path=$(pwd)
base_path=$base_path"/../exp_results"
layer_num=(2 3 4)
kernel_num=(512 1024)
kernel_size=(3 5)
last_kernel_num=(512 1024 2048)
last_kernel_size=(1 2)

counter=0
export CUDA_VISIBLE_DEVICE=0
while true; do
    counter=$((counter+1))
    out_dir=$base_path'/'$counter
    command="\
        python conv_net_sorted.py \
        -out-dir $out_dir \
        -layer-num ${layer_num[$((RANDOM % ${#layer_num[@]}))]} \
        -kernel-size ${kernel_size[$((RANDOM % ${#kernel_size[@]}))]} \
        -kernel-num  ${kernel_num[$((RANDOM % ${#kernel_num[@]}))]} \
        -last-kernel-size  ${last_kernel_size[$((RANDOM % ${#last_kernel_size[@]}))]} \
        -last-kernel-num ${last_kernel_num[$((RANDOM % ${#last_kernel_num[@]}))]} \
        -l2 0 \
        -epochs-num 5 \
        -step-size 0.001"
    echo -e $command
    $command
done
