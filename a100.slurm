#!/bin/bash
#SBATCH -J cond_contrast
#SBATCH -t 120:0:0
#SBATCH -p tier3
#SBATCH -A puzzle
#SBATCH --mem=100g
#SBATCH --mail-user=jcl3689@rit.edu
#SBATCH --gres=gpu:a100:4

#SBATCH -n 1
#SBATCH -c 10

# spack load gcc@9.3.0/hufzekv
# spack load cuda@11.0.2/lrd2rcw
# conda activate openmmlab
# srun -A puzzle -p tier3 --nodes=1 --ntasks-per-node=4 --cpus-per-task=5 --gres=gpu:a100:4 --mem=100g  --pty bash
# srun -J test -A puzzle -p tier3 -t 45:0:0 --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 --gres=gpu:a100:4 --mem=100g  --pty bash
# srun -J test -A puzzle -p tier3 -t 120:0:0 --nodes=1 --ntasks-per-node=4 --cpus-per-task=5 --gres=gpu:a100:4 --mem=100g  --pty bash
# srun -J test -A puzzle -p tier3 -t 120:0:0 --nodes=1 --ntasks-per-node=4 --cpus-per-task=5 --gres=gpu:v100:4 --mem=100g  --pty bash
# conda env list
# NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 GPUS=8 GPUS_PER_NODE=4 ./slurm_train.sh tier3 mae mae_vit_large_patch16 
##### srun -n 1 -c 20 --mem=100g --account=puzzle -p tier3 --gres=gpu:a100:4 -t 10:0:0 --pty bash
# python submitit_pretrain.py --model mae_vit_large_patch16 --job_dir logs/submitit --job_name mae --nodes 4
# python submitit_pretrain.py --model mae_vit_large_patch16 --resume logs/submitit/checkpoint-180.pth --job_dir logs/submitit --job_name mae --nodes 4
python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py --resume logs/submitit/checkpoint-180.pth
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py  --world_size 4 --resume logs/submitit/checkpoint-180.pth
# python submitit_pretrain.py --model mae_vit_large_patch16 --resume logs/checkpoint-200.pth --job_dir logs/submitit --job_name mae --nodes 4





