#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=50
#SBATCH -A RSC_MGT_WI-ST-SCCS-LJ7UAANSP4Q-DEFAULT-CPU
#SBATCH --time=1-00:00:00
#SBATCH --job-name train_ConcatGridClean
#SBATCH --output lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/Noise_Tolerant/Train/Train_GRID/Clean/GridClean.log
#SBATCH --error  lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/Noise_Tolerant/Train/Train_GRID/Clean/ErrorGridClean.log



# Source the bashrc file
source ~/.bashrc

# Load modules

module load cuDNN/8.1.1.33-CUDA-11.2.1
conda activate speaker_identification
module load Anaconda3/2020.11



# Execute the Python script
python lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/Noise_Tolerant/Train/Train_GRID/Clean/train_Concat.py
