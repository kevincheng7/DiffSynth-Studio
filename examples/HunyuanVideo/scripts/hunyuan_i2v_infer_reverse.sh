#!/bin/bash
#SBATCH -o job.%j_hunyuanvideo_i2v_infer_reverse.out
#SBATCH --partition=GPUA800
#SBATCH -J hunyuan_i2v_infer_reverse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


##### Number of total processes
echo "--------------------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:  $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total number of tasks: $SLURM_NTASKS"
echo "Number of tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "--------------------------------------------------------------------------"

nvidia-smi
source /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
conda activate /gpfs/share/home/2201111701/miniconda3/envs/Hunyuan

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo

cd ~/Kaiwencheng/DiffSynth-Studio
python examples/HunyuanVideo/hunyuanvideo_i2v_80G_reverse.py