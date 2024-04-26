#! /bin/bash

SAMPLE_RATES=(8000 12000 16000 24000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python compute_metrics.py -sr $SR -tsr 48000
done