#!/bin/bash
# set up SBATCH args
#SBATCH --job-name=combined_16_3_weight
#SBATCH --partition=gpu_4_a100   # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --ntasks=64
#SBATCH --mem=256G        # self-explanatory, set to your preference
#SBATCH --gres=gpu:4             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference
#SBATCH --nodes=1
#SBATCH --mail-user=pranav.harer@uni-konstanz.de
#SBATCH --qos=normal                  # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2
#SBATCH --time=0-4:00:00


TF_GPU_ALLOCATOR=cuda_malloc_async
TF_ENABLE_ONEDNN_OPTS=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load devel/cuda/12.4
module load devel/python/3.11.7_intel_2021.4.0
source  /home/kn/kn_kn/kn_pop542099/myenv/bin/activate
cd ScriptWorld/Script_World/envs/  
python /home/kn/kn_kn/kn_pop542099/ScriptWorld/Script_World/envs/ppoagent.py #24147455,24163087,24208658


  salloc --job-name = cmagppo -t=4:00:00 -p=gpu_4_a100 --gres=gpu:4 