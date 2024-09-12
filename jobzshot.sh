#!/bin/bash
# set up SBATCH args
#SBATCH --job-name=combined_16_3_weight
#SBATCH --partition=gpu_4_a100   # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --ntasks=64
#SBATCH --mem=256G        # self-explanatory, set to your preference
#SBATCH --gres=gpu:4             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference
#SBATCH --nodes=9
#SBATCH --qos=normal                  # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2
#SBATCH --time=0-48:00:00         # running time limit, 0 as unlimited
TF_GPU_ALLOCATOR=cuda_malloc_async
TF_ENABLE_ONEDNN_OPTS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load devel/cuda/12.4
module load devel/python/3.11.7_intel_2021.4.0
source myenv/bin/activate  
python code/zshot.py #24147305
#cd /home/kn/kn_kn/kn_pop542099/ScriptWorld/Script_World/envs
#python Script_World_terminal.py --scn 'baking a cake' --no_of_actions 2 --allowed_wrong_actions 5  --hop 1 --disclose_state_node --seed 42 --history
#ppotry 23612747(llama2) 23612748(3 - 80)  23612749 (instruct)
#https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoid3o3MXJuM29zcjY5MXRyczE2bjRwMDNnIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNDI0NDk5M319fV19&Signature=KpUegvBNp4%7EYTRl6lh2Bio2FkJYUATT6EAvyHdEeVyvXJFvpm4Jw8tQUHf%7EiK5YUWu8Aojp2dQUFPETUaxxwHIh2XuyJ1P3yZG0B4bZZCq8Uqp8MeBRBE3X94sNURayp7RciARWFm%7Et%7EwjKex7-1lW6-H2l0xb-KLXQAUa7-6YA4bzZ7A6%7Eyy2O4DvT%7EpWQzyI%7EO6YrLYI%7E6gEdzNlrk0r5nF2AnSWchW11KKG0pISnjmPZm9zua5aGoi0M1QZejvpGr6q6iWXb05NZ4tJY74s0LHr90loTCUkh3Jeyg9w2gs55oOoxF1oXgPei5dz0AOTBMxcSurxc2P6%7EXoQX8EQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1301725107471286