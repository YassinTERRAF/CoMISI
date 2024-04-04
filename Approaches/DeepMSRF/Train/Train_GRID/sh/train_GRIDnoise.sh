#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=50
#SBATCH -A RSC_MGT_WI-ST-SCCS-LJ7UAANSP4Q-DEFAULT-CPU
#SBATCH --time=1-00:00:00
#SBATCH --job-name train_GRIDnoise
#SBATCH --output /home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/DeepMSRF/Train/train_GRIDnoise.log
#SBATCH --error  /home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/DeepMSRF/Train/Errortrain_GRIDnoise.log



# Source the bashrc file
source ~/.bashrc

# Load modules

module load cuDNN/8.1.1.33-CUDA-11.2.1
conda activate speaker_identification
module load Anaconda3/2020.11



# Execute the Python script
python lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/DeepMSRF/Train/Train_GRID/Noisy/train_Concat.py

