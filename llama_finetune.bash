#!/bin/bash
export LD_LIBRARY_PATH=/home/miniconda3/envs/HF/lib/python3.7/.../nvidia/cublas/lib/:$LD_LIB
export CUDA_VISIBLE_DEVICES=0,1,2,3  # will use 4 GPUs
###############################
python llama_finetune.py