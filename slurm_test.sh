#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

SAMPLE_RATES=(8000 12000 16000 24000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    MODEL_NAME=pt_rd_80_ft_ub_mrv2
    DATA_DIR=./dataset/testset/source_wavs/$SR
    OUTPUT_DIR=./outputs/48000/$SR
    EXT=wav
    CUDA_VISIBLE_DEVICES=0 python inference_from_audio.py -m $MODEL_NAME -d $DATA_DIR -o $OUTPUT_DIR -e $EXT
done