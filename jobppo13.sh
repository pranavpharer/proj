#!/bin/bash
# set up SBATCH args
#SBATCH --job-name=combined_16_3_weight
#SBATCH --partition=gpu_4_a100   # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --ntasks=64
#SBATCH --mem=256G        # self-explanatory, set to your preference
#SBATCH --gres=gpu:4             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference
#SBATCH --nodes=5
#SBATCH --qos=normal                  # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2
#SBATCH --time=0-48:00:00         # running time limit, 0 as unlimited
TF_GPU_ALLOCATOR=cuda_malloc_async
TF_ENABLE_ONEDNN_OPTS=0
TORCH_USE_CUDA_DSA=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load devel/cuda/12.4 
module load devel/python/3.11.7_intel_2021.4.0
source myenv/bin/activate  

python code/ppo13.py #24010012,TORCH_USE_CUDA_DSA

#cd /home/kn/kn_kn/kn_pop542099/ScriptWorld/Script_World/envs
#python Script_World_terminal.py --scn 'baking a cake' --no_of_actions 2 --allowed_wrong_actions 5  --hop 1 --disclose_state_node --seed 42 --history
#ppotry 23612747(llama2) 23612748(3 - 80)  23612749 (instruct)TORCH_USE_CUDA_DSA